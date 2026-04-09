"""
Publisher: moves passing articles to the HuggingFace dataset exporter.

Thin coordination layer — does not contain export logic (that lives in
src/training_data/exporter.py). Responsible for:
- Checking aha_score.passed before forwarding
- Marking articles as published
- Returning summary counts
"""

from dataclasses import dataclass

from .generator import GeneratedArticle
from ..training_data.exporter import DatasetExporter


@dataclass
class PublishResult:
    published: int
    skipped: int
    errors: int


class ArticlePublisher:
    """
    Gate between generator output and dataset exporter.

    Never forwards an article with aha_score.passed=False.
    """

    def __init__(self, exporter: DatasetExporter):
        self.exporter = exporter

    def publish_batch(
        self, articles: list[GeneratedArticle]
    ) -> PublishResult:
        published = 0
        skipped = 0
        errors = 0

        for article in articles:
            if not article.aha_score.passed:
                skipped += 1
                continue
            try:
                record_id = self.exporter.append(article)
                article.published = True
                article.huggingface_id = record_id
                published += 1
            except Exception:
                errors += 1

        return PublishResult(published=published, skipped=skipped, errors=errors)
