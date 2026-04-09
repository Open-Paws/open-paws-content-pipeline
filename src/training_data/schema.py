"""
Dataset schema for Open Paws animal welfare training data on HuggingFace.

Target dataset: open-paws/animal-welfare-training-data

Every record carries enough metadata for downstream researchers to:
- Filter by language, topic category, or AHA score component
- Identify which model generated and evaluated the content
- Reproduce the generation conditions
- Filter out low-confidence records

The schema is intentionally flat (no nested objects) for maximum
compatibility with HuggingFace datasets loading patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TrainingDataRecord:
    """One record in the HuggingFace training dataset."""

    # Content
    text: str              # The article body
    title: str
    topic: str             # The topic seed string used for generation

    # Language
    language: str          # ISO 639-1 code ("en", "hi", "es", "fr", "zh")

    # Stats
    word_count: int

    # AHA evaluation
    aha_accurate: float
    aha_helpful: float
    aha_animal_positive: float
    aha_composite: float
    aha_passed: bool
    aha_flags: str         # JSON-encoded list; HF datasets prefers flat strings

    # Provenance
    generated_date: str    # ISO 8601
    generator_model: str   # e.g. "claude-haiku-4-5-20251001"
    evaluator_model: str   # e.g. "claude-sonnet-4-6"

    # Movement metadata
    topic_category: str           # "factory_farming", "legislation", "science", etc.
    species_focus: str            # JSON-encoded list: ["chickens", "pigs"]
    action_oriented: bool         # Does the article call for specific action?

    @classmethod
    def from_generated_article(
        cls,
        article,  # GeneratedArticle — avoid circular import
        topic_category: str = "general",
        species_focus: list[str] | None = None,
        action_oriented: bool = False,
        generator_model: str = "claude-haiku-4-5-20251001",
        evaluator_model: str = "claude-sonnet-4-6",
    ) -> "TrainingDataRecord":
        import json

        score = article.aha_score
        return cls(
            text=article.body,
            title=article.title,
            topic=article.topic,
            language=article.language,
            word_count=article.word_count,
            aha_accurate=score.accurate,
            aha_helpful=score.helpful,
            aha_animal_positive=score.animal_positive,
            aha_composite=score.composite,
            aha_passed=score.passed,
            aha_flags=json.dumps(score.flags),
            generated_date=datetime.now(timezone.utc).isoformat(),
            generator_model=generator_model,
            evaluator_model=evaluator_model,
            topic_category=topic_category,
            species_focus=json.dumps(species_focus or []),
            action_oriented=action_oriented,
        )

    def to_dict(self) -> dict:
        """Flat dict for JSONL export and HuggingFace datasets."""
        return {
            "text": self.text,
            "title": self.title,
            "topic": self.topic,
            "language": self.language,
            "word_count": self.word_count,
            "aha_accurate": self.aha_accurate,
            "aha_helpful": self.aha_helpful,
            "aha_animal_positive": self.aha_animal_positive,
            "aha_composite": self.aha_composite,
            "aha_passed": self.aha_passed,
            "aha_flags": self.aha_flags,
            "generated_date": self.generated_date,
            "generator_model": self.generator_model,
            "evaluator_model": self.evaluator_model,
            "topic_category": self.topic_category,
            "species_focus": self.species_focus,
            "action_oriented": self.action_oriented,
        }
