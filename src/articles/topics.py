"""
Topic management for the article generation pipeline.

Loads topics from config/topics.yaml and provides:
- random_topics(n): sample n topics across categories
- topics_by_category(cat): all topics in one category
- expand_topic(topic): use LLM to generate angle variations on a topic

All topics use movement terminology: "farmed animals", "factory farm", "slaughterhouse".
"""

import os
import random
from pathlib import Path
from typing import Optional

import yaml


_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "topics.yaml"


def _load_topics(path: Path = _DEFAULT_CONFIG) -> dict[str, list[str]]:
    with open(path) as f:
        return yaml.safe_load(f)


def topics_by_category(category: str, config_path: Optional[Path] = None) -> list[str]:
    """Return all topics for a given category key."""
    data = _load_topics(config_path or _DEFAULT_CONFIG)
    return data.get(category, [])


def all_categories(config_path: Optional[Path] = None) -> list[str]:
    """Return all category keys."""
    return list(_load_topics(config_path or _DEFAULT_CONFIG).keys())


def all_topics(config_path: Optional[Path] = None) -> list[str]:
    """Return flat list of all topics across all categories."""
    data = _load_topics(config_path or _DEFAULT_CONFIG)
    return [topic for topics in data.values() for topic in topics]


def random_topics(n: int, config_path: Optional[Path] = None) -> list[str]:
    """
    Sample n topics with category-balanced distribution.

    Cycles through categories to avoid over-representing any single one.
    If n exceeds total topic count, topics repeat (with shuffle between cycles).
    """
    data = _load_topics(config_path or _DEFAULT_CONFIG)
    categories = list(data.keys())
    result: list[str] = []

    while len(result) < n:
        for cat in categories:
            if len(result) >= n:
                break
            pool = data[cat]
            result.append(random.choice(pool))

    return result[:n]


class TopicSeed:
    """
    Static interface used by ArticleGenerator and the orchestrator.

    Wraps module-level functions so callers don't need to manage config path.
    """

    @staticmethod
    def random_topics(n: int) -> list[str]:
        return random_topics(n)

    @staticmethod
    def all_topics() -> list[str]:
        return all_topics()

    @staticmethod
    def by_category(category: str) -> list[str]:
        return topics_by_category(category)


def expand_topic(topic: str, n_angles: int = 3) -> list[str]:
    """
    Use LLM to generate n angle variations on a topic.

    Returns a list of refined topic strings suitable for article generation.
    Falls back to [topic] on any error so callers always get at least one result.
    """
    import anthropic

    prompt = (
        f"Generate {n_angles} distinct article angles on this animal advocacy topic:\n"
        f"{topic}\n\n"
        "Rules:\n"
        "- Use 'farmed animals' not 'livestock'\n"
        "- Use 'factory farm' not 'farm' or 'agricultural operation'\n"
        "- Use 'slaughterhouse' not 'processing plant'\n"
        "- Each angle should be a single sentence topic statement\n"
        "- Vary the angle: policy, science, corporate, investigation, movement\n\n"
        "Respond with a JSON array of strings only, no markdown:\n"
        '["angle 1", "angle 2", "angle 3"]'
    )

    try:
        client = anthropic.Anthropic()
        model = os.getenv("TOPIC_EXPAND_MODEL", "claude-haiku-4-5-20251001")
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        angles = json.loads(response.content[0].text.strip())
        return [str(a) for a in angles]
    except Exception:
        return [topic]
