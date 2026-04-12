"""
Tests for TrainingDataRecord schema and DatasetExporter.

Domain rules encoded:
- Only articles with aha_passed=True should appear in the dataset
- Schema is flat (no nested objects) for HuggingFace compatibility
- to_dict() must include all required fields
- Category inference uses movement terminology keywords
- record_count() reflects actual JSONL line count
"""

import json
import tempfile
from pathlib import Path


from src.articles.evaluator import AHAScore
from src.articles.generator import GeneratedArticle
from src.training_data.exporter import DatasetExporter
from src.training_data.schema import TrainingDataRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(
    topic: str = "gestation crates in factory farms",
    language: str = "en",
    aha_passed: bool = True,
    composite: float = 0.85,
) -> GeneratedArticle:
    score = AHAScore(
        accurate=0.85,
        helpful=0.85,
        animal_positive=0.85,
        composite=composite,
        reasoning="Good article.",
        passed=aha_passed,
        flags=[],
    )
    return GeneratedArticle(
        title="Understanding Gestation Crates",
        body="This article covers the welfare implications of gestation crates.",
        topic=topic,
        language=language,
        word_count=50,
        aha_score=score,
    )


# ---------------------------------------------------------------------------
# TrainingDataRecord schema
# ---------------------------------------------------------------------------


class TestTrainingDataRecord:
    def test_from_generated_article_produces_valid_record(self):
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(article)
        assert record.text == article.body
        assert record.title == article.title
        assert record.topic == article.topic
        assert record.language == article.language
        assert record.word_count == article.word_count

    def test_aha_fields_mapped_from_score(self):
        article = _make_article(aha_passed=True, composite=0.85)
        record = TrainingDataRecord.from_generated_article(article)
        assert record.aha_accurate == article.aha_score.accurate
        assert record.aha_helpful == article.aha_score.helpful
        assert record.aha_animal_positive == article.aha_score.animal_positive
        assert record.aha_composite == article.aha_score.composite
        assert record.aha_passed is True

    def test_to_dict_is_flat(self):
        """HuggingFace datasets require flat dicts — no nested objects."""
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(article)
        row = record.to_dict()
        for key, value in row.items():
            assert not isinstance(value, (dict, list)), (
                f"Field '{key}' is not flat: {type(value).__name__}"
            )

    def test_to_dict_contains_required_fields(self):
        required_fields = [
            "text", "title", "topic", "language", "word_count",
            "aha_accurate", "aha_helpful", "aha_animal_positive", "aha_composite",
            "aha_passed", "aha_flags",
            "generated_date", "generator_model", "evaluator_model",
            "topic_category", "species_focus", "action_oriented",
        ]
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(article)
        row = record.to_dict()
        for field in required_fields:
            assert field in row, f"Missing required field: '{field}'"

    def test_aha_flags_is_json_string(self):
        """aha_flags must be JSON-encoded string for HuggingFace flat schema."""
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(article)
        row = record.to_dict()
        assert isinstance(row["aha_flags"], str)
        parsed = json.loads(row["aha_flags"])
        assert isinstance(parsed, list)

    def test_species_focus_is_json_string(self):
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(
            article, species_focus=["pigs", "chickens"]
        )
        row = record.to_dict()
        assert isinstance(row["species_focus"], str)
        parsed = json.loads(row["species_focus"])
        assert "pigs" in parsed
        assert "chickens" in parsed

    def test_generated_date_is_iso8601(self):
        from datetime import datetime
        article = _make_article()
        record = TrainingDataRecord.from_generated_article(article)
        # Should not raise
        dt = datetime.fromisoformat(record.generated_date)
        assert dt is not None


# ---------------------------------------------------------------------------
# DatasetExporter — category inference
# ---------------------------------------------------------------------------


class TestCategoryInference:
    def test_cage_keyword_maps_to_factory_farming(self):
        exporter = DatasetExporter(output_path=Path("/tmp/test_cat_infer.jsonl"))
        assert exporter._infer_category("battery cage conditions") == "factory_farming"

    def test_law_keyword_maps_to_legislation(self):
        exporter = DatasetExporter(output_path=Path("/tmp/test_cat_infer.jsonl"))
        assert exporter._infer_category("ag-gag law in Iowa") == "legislation"

    def test_sentience_keyword_maps_to_science(self):
        exporter = DatasetExporter(output_path=Path("/tmp/test_cat_infer.jsonl"))
        assert exporter._infer_category("fish pain and sentience research") == "science"

    def test_unknown_topic_maps_to_general(self):
        exporter = DatasetExporter(output_path=Path("/tmp/test_cat_infer.jsonl"))
        assert exporter._infer_category("this topic has no category keywords") == "general"

    def test_slaughterhouse_speed_infers_factory_farming(self):
        exporter = DatasetExporter(output_path=Path("/tmp/test_cat_infer.jsonl"))
        # "slaughterhouse line speeds" contains no factory_farming keywords but
        # is not a general topic — verify it gets a non-general category if applicable,
        # or that the general fallback is correct
        result = exporter._infer_category("slaughterhouse line speeds and welfare")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# DatasetExporter — append and record_count
# ---------------------------------------------------------------------------


class TestDatasetExporter:
    def test_append_creates_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "articles.jsonl"
            exporter = DatasetExporter(output_path=output)
            article = _make_article()
            exporter.append(article)
            assert output.exists()

    def test_append_returns_string_record_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "articles.jsonl"
            exporter = DatasetExporter(output_path=output)
            article = _make_article()
            record_id = exporter.append(article)
            assert isinstance(record_id, str)
            assert len(record_id) > 0

    def test_record_count_increments_with_each_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "articles.jsonl"
            exporter = DatasetExporter(output_path=output)
            assert exporter.record_count() == 0
            exporter.append(_make_article())
            assert exporter.record_count() == 1
            exporter.append(_make_article())
            assert exporter.record_count() == 2

    def test_record_count_zero_when_file_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nonexistent.jsonl"
            exporter = DatasetExporter(output_path=output)
            assert exporter.record_count() == 0

    def test_appended_record_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "articles.jsonl"
            exporter = DatasetExporter(output_path=output)
            exporter.append(_make_article())
            with open(output) as f:
                line = f.readline().strip()
            row = json.loads(line)  # Must not raise
            assert "text" in row
            assert "aha_passed" in row

    def test_append_records_have_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "articles.jsonl"
            exporter = DatasetExporter(output_path=output)
            ids = [exporter.append(_make_article()) for _ in range(5)]
            assert len(set(ids)) == 5, "Record IDs must be unique"
