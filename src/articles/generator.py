"""
Article generator with NAV language gate and mandatory AHA evaluation gate.

Uses cheapest capable model (claude-haiku) for generation, routed through
the Open Paws API gateway for centralised cost tracking and key management.

Generation flow:
  1. Receive topic string
  2. POST to gateway /claude/messages with advocacy-aware system prompt
  3. NAV language check (mcp-server-nav-language) — blocks on ERROR violations;
     service errors are fail-open (NAV outage does not block the pipeline).
  4. Pass output to AHAEvaluator
  5. Return GeneratedArticle with aha_score.passed indicating publishability
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from .evaluator import AHAEvaluator, AHAScore
from .nav_checker import check_article_language
from .topics import TopicSeed

logger = logging.getLogger(__name__)


@dataclass
class GeneratedArticle:
    title: str
    body: str
    topic: str
    language: str
    word_count: int
    aha_score: AHAScore
    published: bool = False
    huggingface_id: Optional[str] = None


ARTICLE_SYSTEM_PROMPT = """You are writing for Open Paws, an animal liberation platform.
Write accurate, compelling articles about animal welfare, factory farming, and advocacy.

Mandatory language rules:
- Use "farmed animals" not "livestock" or "farm animals"
- Use "factory farm" not "farm", "ranch", or "agricultural operation"
- Use "slaughterhouse" not "processing plant", "packing plant", or "processing facility"
- Use "gestation crates" not "farrowing crates" when referring to pre-birth confinement
- Center the animal experience — they are sentient individuals with interests, not units

Content rules:
- Be factually rigorous: cite specific research, regulatory data, or documented investigations
  when making empirical claims. Do not invent statistics.
- Write for a general audience, not just existing advocates
- Length: 600–900 words
- Structure: title on first line (no # prefix), then article body
- Open with a specific, concrete detail — not a rhetorical question or vague generalization

Never: normalize exploitation, use industry euphemisms, write in a way that treats
animal bodies as products."""


class ArticleGenerator:
    """
    Generate articles and gate them through AHA evaluation.

    Single responsibility: call the generation model via the Open Paws API
    gateway, call the evaluator, return a GeneratedArticle. Does not write
    to disk or publish.
    """

    def __init__(self, threshold: float = 0.75):
        self.gateway_url = os.environ.get(
            "OPEN_PAWS_GATEWAY_URL", "https://api.openpaws.ai/v1"
        ).rstrip("/")
        self.api_key = os.environ.get("OPEN_PAWS_API_KEY", "")
        self.model = os.environ.get("ARTICLE_GEN_MODEL", "claude-haiku-4-5-20251001")
        self.evaluator = AHAEvaluator(threshold=threshold)

    def generate(self, topic: str, angle: str = "") -> Optional[GeneratedArticle]:
        """
        Generate and evaluate one article.

        Returns None only if the API call itself fails.
        Returns GeneratedArticle with aha_score.passed=False if it fails the gate.
        Callers must check aha_score.passed before publishing.
        """
        prompt = f"Write an article about: {topic}"
        if angle:
            prompt += f"\nAngle: {angle}"

        try:
            resp = httpx.post(
                f"{self.gateway_url}/claude/messages",
                json={
                    "model": self.model,
                    "max_tokens": 1500,
                    "system": ARTICLE_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            body = data["content"][0]["text"]
        except Exception:
            return None

        # Title is first line; strip any leading # characters
        lines = body.splitlines()
        title = lines[0].lstrip("#").strip() if lines else topic

        # NAV language gate — runs before AHA to catch speciesist language early.
        # Service errors are fail-open (AHA gate still runs); ERROR violations block.
        nav_result = check_article_language(body, article_id=topic)
        if nav_result.get("service_error"):
            logger.warning(
                "NAV check skipped (service error: %s) — continuing to AHA gate",
                nav_result["service_error"],
            )
        elif not nav_result["is_clean"]:
            logger.warning(
                "Article BLOCKED — %d speciesist language violation(s) detected. "
                "Matched: %s | Topic: %s",
                nav_result["error_count"],
                [
                    v.get("matched_text", "<unknown>")
                    for v in nav_result["violations"]
                    if v.get("severity") == "error"
                ],
                topic,
            )
            return GeneratedArticle(
                title=title,
                body=body,
                topic=topic,
                language="en",
                word_count=len(body.split()),
                aha_score=AHAScore(
                    accurate=0.0,
                    helpful=0.0,
                    animal_positive=0.0,
                    composite=0.0,
                    reasoning="Blocked by NAV language gate before AHA evaluation.",
                    passed=False,
                    flags=[f"NAV_BLOCKED:{nav_result['error_count']}_errors"],
                ),
            )

        aha = self.evaluator.evaluate(body, title)

        return GeneratedArticle(
            title=title,
            body=body,
            topic=topic,
            language="en",
            word_count=len(body.split()),
            aha_score=aha,
            published=False,
        )

    def generate_batch(
        self, count: int
    ) -> tuple[list[GeneratedArticle], list[GeneratedArticle]]:
        """
        Generate count articles using category-balanced topic sampling.

        Returns (passed, failed) tuple.
        passed: articles that cleared the AHA gate (safe to publish)
        failed: articles below threshold (need human review)
        """
        topics = TopicSeed.random_topics(count)
        passed: list[GeneratedArticle] = []
        failed: list[GeneratedArticle] = []

        for topic in topics:
            article = self.generate(topic)
            if article is None:
                continue
            if article.aha_score.passed:
                passed.append(article)
            else:
                failed.append(article)

        return passed, failed
