"""
Tests for ArticlePublisher — the gate between generator output and dataset export.

Domain rules encoded:
- Publisher MUST NEVER forward an article with aha_score.passed=False
- Publisher marks forwarded articles as published
- Publisher tracks published/skipped/error counts accurately
- Exporter exceptions are caught per-article (one error must not stop the batch)
"""

from unittest.mock import MagicMock


from src.articles.evaluator import AHAScore
from src.articles.generator import GeneratedArticle
from src.articles.publisher import ArticlePublisher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(passed: bool) -> GeneratedArticle:
    score = AHAScore(
        accurate=0.9 if passed else 0.3,
        helpful=0.9 if passed else 0.3,
        animal_positive=0.9 if passed else 0.3,
        composite=0.9 if passed else 0.3,
        reasoning="Test.",
        passed=passed,
        flags=[],
    )
    return GeneratedArticle(
        title="Test Article on Farmed Animal Welfare",
        body="Article body discussing factory farm conditions.",
        topic="factory farming",
        language="en",
        word_count=120,
        aha_score=score,
    )


def _make_publisher(exporter_raises: bool = False) -> tuple[ArticlePublisher, MagicMock]:
    mock_exporter = MagicMock()
    if exporter_raises:
        mock_exporter.append.side_effect = IOError("Write failed")
    else:
        mock_exporter.append.return_value = "record-uuid-abc123"
    return ArticlePublisher(exporter=mock_exporter), mock_exporter


# ---------------------------------------------------------------------------
# Publication gate — never publish failed articles
# ---------------------------------------------------------------------------


class TestPublicationGate:
    def test_failing_article_is_skipped_not_published(self):
        """CRITICAL: aha_score.passed=False articles must never reach the exporter."""
        publisher, mock_exporter = _make_publisher()
        article = _make_article(passed=False)
        result = publisher.publish_batch([article])
        mock_exporter.append.assert_not_called()
        assert result.skipped == 1
        assert result.published == 0

    def test_passing_article_is_forwarded_to_exporter(self):
        publisher, mock_exporter = _make_publisher()
        article = _make_article(passed=True)
        result = publisher.publish_batch([article])
        mock_exporter.append.assert_called_once_with(article)
        assert result.published == 1
        assert result.skipped == 0

    def test_mixed_batch_only_forwards_passing_articles(self):
        publisher, mock_exporter = _make_publisher()
        passing = [_make_article(passed=True) for _ in range(3)]
        failing = [_make_article(passed=False) for _ in range(2)]
        articles = passing + failing
        result = publisher.publish_batch(articles)
        assert result.published == 3
        assert result.skipped == 2
        assert mock_exporter.append.call_count == 3

    def test_empty_batch_returns_zero_counts(self):
        publisher, _ = _make_publisher()
        result = publisher.publish_batch([])
        assert result.published == 0
        assert result.skipped == 0
        assert result.errors == 0


# ---------------------------------------------------------------------------
# Article state mutation on publish
# ---------------------------------------------------------------------------


class TestArticleStateMutation:
    def test_published_flag_is_set_after_export(self):
        publisher, _ = _make_publisher()
        article = _make_article(passed=True)
        assert article.published is False
        publisher.publish_batch([article])
        assert article.published is True

    def test_huggingface_id_is_set_after_export(self):
        publisher, _ = _make_publisher()
        article = _make_article(passed=True)
        assert article.huggingface_id is None
        publisher.publish_batch([article])
        assert article.huggingface_id == "record-uuid-abc123"

    def test_failed_article_published_flag_stays_false(self):
        publisher, _ = _make_publisher()
        article = _make_article(passed=False)
        publisher.publish_batch([article])
        assert article.published is False


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    def test_exporter_error_counted_not_raised(self):
        """One export error must not kill the rest of the batch."""
        publisher, _ = _make_publisher(exporter_raises=True)
        article = _make_article(passed=True)
        result = publisher.publish_batch([article])
        assert result.errors == 1
        assert result.published == 0

    def test_one_error_does_not_stop_subsequent_articles(self):
        """Exporter raises on first call only — second article must still publish."""
        mock_exporter = MagicMock()
        mock_exporter.append.side_effect = [IOError("Disk full"), "record-id-2"]
        publisher = ArticlePublisher(exporter=mock_exporter)
        articles = [_make_article(passed=True), _make_article(passed=True)]
        result = publisher.publish_batch(articles)
        assert result.errors == 1
        assert result.published == 1
