"""
Tests for pipeline orchestrator.

Domain rules encoded:
- Cost estimates are computed per article (generation + evaluation)
- PipelineStats.summary() includes pass rate, cost, avg AHA score
- Pipeline dry-run mode never calls exporter.push()
- aha_passed/aha_failed counts are accurate
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.orchestrator import (
    BATCH_SIZE,
    COST_PER_ARTICLE_GENERATION,
    COST_PER_AHA_EVALUATION,
    COST_PER_ARTICLE_TOTAL,
    PipelineStats,
)


# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------


class TestCostConstants:
    def test_total_cost_is_sum_of_components(self):
        """Cost per article = generation cost + evaluation cost."""
        assert (
            COST_PER_ARTICLE_TOTAL
            == COST_PER_ARTICLE_GENERATION + COST_PER_AHA_EVALUATION
        )

    def test_cost_per_article_total_under_one_cent(self):
        """At $0.006/article, 1000 articles ~$6 — within $100/month budget."""
        assert COST_PER_ARTICLE_TOTAL <= 0.01, (
            f"Cost per article {COST_PER_ARTICLE_TOTAL} exceeds $0.01 target"
        )

    def test_batch_size_is_positive_integer(self):
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0


# ---------------------------------------------------------------------------
# PipelineStats
# ---------------------------------------------------------------------------


class TestPipelineStats:
    def test_default_stats_are_zero(self):
        stats = PipelineStats()
        assert stats.total_attempted == 0
        assert stats.aha_passed == 0
        assert stats.aha_failed == 0
        assert stats.published == 0
        assert stats.estimated_cost_usd == 0.0

    def test_summary_includes_pass_rate(self):
        stats = PipelineStats(
            total_attempted=10,
            aha_passed=8,
            aha_failed=2,
            published=8,
            estimated_cost_usd=0.06,
            avg_aha_score=0.88,
            elapsed_seconds=12.5,
        )
        summary = stats.summary()
        assert "80%" in summary  # 8/(8+2) = 80%

    def test_summary_includes_cost(self):
        stats = PipelineStats(estimated_cost_usd=0.0600)
        summary = stats.summary()
        assert "0.0600" in summary

    def test_summary_with_zero_articles_does_not_divide_by_zero(self):
        stats = PipelineStats()
        # Must not raise ZeroDivisionError
        summary = stats.summary()
        assert "Pipeline" in summary

    def test_summary_includes_aha_counts(self):
        stats = PipelineStats(aha_passed=7, aha_failed=3)
        summary = stats.summary()
        assert "7" in summary
        assert "3" in summary


# ---------------------------------------------------------------------------
# Helpers for building mock articles
# ---------------------------------------------------------------------------


def _make_mock_article(passed: bool):
    from src.articles.evaluator import AHAScore
    from src.articles.generator import GeneratedArticle

    score = AHAScore(
        accurate=0.9,
        helpful=0.9,
        animal_positive=0.9,
        composite=0.9 if passed else 0.4,
        reasoning=".",
        passed=passed,
        flags=[],
    )
    return GeneratedArticle(
        title="T",
        body="B",
        topic="t",
        language="en",
        word_count=10,
        aha_score=score,
    )


# ---------------------------------------------------------------------------
# run_pipeline — dry-run mode
#
# ArticleGenerator and DatasetExporter are imported inside run_pipeline()
# via local imports, so they must be patched at their source module paths.
# ---------------------------------------------------------------------------


class TestRunPipelineDryRun:
    def test_dry_run_does_not_push_to_huggingface(self):
        """dry_run=True must never call exporter.push()."""
        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = (
            [_make_mock_article(True)],
            [],
        )
        mock_exporter = MagicMock()

        with (
            patch(
                "src.articles.generator.ArticleGenerator",
                return_value=mock_generator,
            ),
            patch(
                "src.training_data.exporter.DatasetExporter",
                return_value=mock_exporter,
            ),
        ):
            from src.pipeline.orchestrator import run_pipeline

            stats = run_pipeline(
                count=1,
                dry_run=True,
                publish=False,
                output_path=Path("/tmp/test_dry_run.jsonl"),
            )

        mock_exporter.push.assert_not_called()

    def test_stats_track_aha_pass_and_fail_counts(self):
        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = (
            [_make_mock_article(True), _make_mock_article(True)],
            [_make_mock_article(False)],
        )
        mock_exporter = MagicMock()

        with (
            patch(
                "src.articles.generator.ArticleGenerator",
                return_value=mock_generator,
            ),
            patch(
                "src.training_data.exporter.DatasetExporter",
                return_value=mock_exporter,
            ),
        ):
            from src.pipeline.orchestrator import run_pipeline

            stats = run_pipeline(
                count=3,
                dry_run=True,
                publish=False,
                output_path=Path("/tmp/test_counts.jsonl"),
            )

        assert stats.aha_passed == 2
        assert stats.aha_failed == 1

    def test_estimated_cost_computed_from_count(self):
        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = ([], [])
        mock_exporter = MagicMock()

        with (
            patch(
                "src.articles.generator.ArticleGenerator",
                return_value=mock_generator,
            ),
            patch(
                "src.training_data.exporter.DatasetExporter",
                return_value=mock_exporter,
            ),
        ):
            from src.pipeline.orchestrator import run_pipeline

            result = run_pipeline(
                count=10,
                dry_run=True,
                publish=False,
                output_path=Path("/tmp/test_cost.jsonl"),
            )

        expected = 10 * COST_PER_ARTICLE_TOTAL
        assert abs(result.estimated_cost_usd - expected) < 1e-9
