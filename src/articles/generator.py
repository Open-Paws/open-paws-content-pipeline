"""
Article generator with mandatory AHA evaluation gate.

Uses cheapest capable model (claude-haiku) for generation.
No article reaches the publisher without passing AHA evaluation.

Generation flow:
  1. Receive topic string
  2. Call claude-haiku with advocacy-aware system prompt
  3. Pass output to AHAEvaluator
  4. Return GeneratedArticle with aha_score.passed indicating publishability
"""

import os
from dataclasses import dataclass
from typing import Optional

from .client import get_client
from .evaluator import AHAEvaluator, AHAScore
from .topics import TopicSeed


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

    Single responsibility: call the generation model, call the evaluator,
    return a GeneratedArticle. Does not write to disk or publish.
    """

    def __init__(self, threshold: float = 0.75):
        self.client = get_client()
        self.model = os.getenv("ARTICLE_GEN_MODEL", "claude-haiku-4-5-20251001")
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
            response = self.client.create_message(
                model=self.model,
                max_tokens=1500,
                system=ARTICLE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            return None

        body = response.text
        # Title is first line; strip any leading # characters
        lines = body.splitlines()
        title = lines[0].lstrip("#").strip() if lines else topic

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
