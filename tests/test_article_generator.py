"""
Tests for ArticleGenerator — generation + AHA gate integration.

Domain rules encoded:
- Generator uses haiku (cheap) for generation, sonnet (judgment) for evaluation
- Articles failing AHA gate must not be published (aha_score.passed=False)
- Generator returns None only on API failure, not on AHA failure
- Title is extracted from first line of body
- Required fields: title, body, topic, language, word_count
"""

from unittest.mock import MagicMock, patch


from src.articles.client import MessageResponse
from src.articles.evaluator import AHAScore
from src.articles.generator import ArticleGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_passing_aha_score() -> AHAScore:
    return AHAScore(
        accurate=0.9,
        helpful=0.9,
        animal_positive=0.9,
        composite=0.9,
        reasoning="All good.",
        passed=True,
        flags=[],
    )


def _make_failing_aha_score() -> AHAScore:
    return AHAScore(
        accurate=0.4,
        helpful=0.5,
        animal_positive=0.3,
        composite=0.4,
        reasoning="Uses industry framing.",
        passed=False,
        flags=["Uses 'livestock' instead of 'farmed animals'"],
    )


def _make_generator(body_text: str, aha_score: AHAScore) -> ArticleGenerator:
    """Build a generator with mocked client and evaluator."""
    generator = ArticleGenerator.__new__(ArticleGenerator)
    generator.threshold = 0.75
    generator.model = "claude-haiku-4-5-20251001"

    mock_client = MagicMock()
    mock_client.create_message.return_value = MessageResponse(text=body_text)
    generator.client = mock_client

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = aha_score
    generator.evaluator = mock_evaluator

    return generator


# ---------------------------------------------------------------------------
# Required fields on GeneratedArticle
# ---------------------------------------------------------------------------


class TestGeneratedArticleFields:
    def test_article_has_title(self):
        body = "Factory Farms and Farmed Animal Welfare\n\nThis article discusses..."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("factory farming topic")
        assert article is not None
        assert article.title != ""

    def test_article_has_body(self):
        body = "Some Title\n\nArticle body content here."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.body == body

    def test_article_has_topic(self):
        body = "Title\n\nBody content."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("pig cognition and social behavior")
        assert article is not None
        assert article.topic == "pig cognition and social behavior"

    def test_article_language_is_english(self):
        body = "Title\n\nBody content."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.language == "en"

    def test_article_has_word_count(self):
        body = "Title\n\nOne two three four five."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.word_count > 0

    def test_word_count_reflects_body_length(self):
        body = "Title\n\n" + " ".join(["word"] * 100)
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        # word_count = len(body.split()) which includes title words
        assert article.word_count >= 100


# ---------------------------------------------------------------------------
# Title extraction from first line
# ---------------------------------------------------------------------------


class TestTitleExtraction:
    def test_title_is_first_line(self):
        body = "Factory Farming and the Hidden Costs\n\nArticle body."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.title == "Factory Farming and the Hidden Costs"

    def test_title_strips_markdown_heading_prefix(self):
        body = "# The Truth About Slaughterhouses\n\nArticle body."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.title == "The Truth About Slaughterhouses"

    def test_title_strips_multiple_hash_prefix(self):
        body = "## Farmed Animal Sentience\n\nArticle body."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.title == "Farmed Animal Sentience"


# ---------------------------------------------------------------------------
# AHA gate integration
# ---------------------------------------------------------------------------


class TestAHAGateIntegration:
    def test_passing_aha_score_sets_passed_true(self):
        body = "Title\n\nBody content."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.aha_score.passed is True

    def test_failing_aha_score_sets_passed_false(self):
        """Article below 0.75 must have passed=False — it must not be published."""
        body = "Title\n\nBody using 'livestock' and 'processing facility'."
        generator = _make_generator(body, _make_failing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.aha_score.passed is False

    def test_failing_article_is_returned_not_suppressed(self):
        """Generator returns failed articles for review queue — never silently drops them."""
        body = "Title\n\nBody."
        generator = _make_generator(body, _make_failing_aha_score())
        article = generator.generate("topic")
        assert article is not None  # Must return something, not None

    def test_article_starts_unpublished(self):
        body = "Title\n\nBody."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        assert article.published is False

    def test_evaluator_is_called_with_body_and_title(self):
        body = "My Article Title\n\nArticle content."
        generator = _make_generator(body, _make_passing_aha_score())
        article = generator.generate("topic")
        assert article is not None
        generator.evaluator.evaluate.assert_called_once_with(body, "My Article Title")


# ---------------------------------------------------------------------------
# API failure handling
# ---------------------------------------------------------------------------


class TestAPIFailureHandling:
    def test_api_failure_returns_none(self):
        """Generator returns None (not an article) when the API itself fails."""
        generator = ArticleGenerator.__new__(ArticleGenerator)
        generator.threshold = 0.75
        generator.model = "claude-haiku-4-5-20251001"

        mock_client = MagicMock()
        mock_client.create_message.side_effect = ConnectionError("Timeout")
        generator.client = mock_client
        generator.evaluator = MagicMock()

        result = generator.generate("topic")
        assert result is None


# ---------------------------------------------------------------------------
# generate_batch routing
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_batch_returns_passed_and_failed_tuples(self):
        body = "Title\n\nBody."
        with patch("src.articles.generator.TopicSeed") as mock_seed:
            mock_seed.random_topics.return_value = ["topic1", "topic2", "topic3"]

            generator = ArticleGenerator.__new__(ArticleGenerator)
            generator.threshold = 0.75
            generator.model = "claude-haiku-4-5-20251001"
            generator.evaluator = MagicMock()

            # First two calls pass, third fails
            generator.evaluator.evaluate.side_effect = [
                _make_passing_aha_score(),
                _make_passing_aha_score(),
                _make_failing_aha_score(),
            ]

            mock_client = MagicMock()
            mock_client.create_message.return_value = MessageResponse(text=body)
            generator.client = mock_client

            passed, failed = generator.generate_batch(3)

        assert len(passed) == 2
        assert len(failed) == 1
        assert all(a.aha_score.passed for a in passed)
        assert all(not a.aha_score.passed for a in failed)
