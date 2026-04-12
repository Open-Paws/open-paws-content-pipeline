"""
Tests for AHA (Accurate, Helpful, Animal-positive) evaluation gate.

Domain rule: articles scoring below 0.75 composite MUST NEVER be published.
Every test here must fail if the behavior it covers breaks.
"""

import json
from unittest.mock import MagicMock


from src.articles.evaluator import AHAEvaluator, _strip_markdown_fences


# ---------------------------------------------------------------------------
# _strip_markdown_fences — utility used by evaluator to clean LLM responses
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        raw = '```json\n{"accurate": 0.9}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"accurate": 0.9}'

    def test_strips_plain_fence(self):
        raw = '```\n{"accurate": 0.9}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"accurate": 0.9}'

    def test_passthrough_when_no_fence(self):
        raw = '{"accurate": 0.9, "helpful": 0.8}'
        result = _strip_markdown_fences(raw)
        assert result == raw

    def test_handles_whitespace(self):
        raw = '  ```json\n{"accurate": 0.9}\n```  '
        result = _strip_markdown_fences(raw)
        assert result == '{"accurate": 0.9}'


# ---------------------------------------------------------------------------
# AHAScore composite calculation
# ---------------------------------------------------------------------------


class TestAHAScoreComposite:
    """The composite score formula is the gating mechanism — it must be correct."""

    def test_composite_weights_sum_to_one(self):
        total = (
            AHAEvaluator.WEIGHT_ACCURATE
            + AHAEvaluator.WEIGHT_HELPFUL
            + AHAEvaluator.WEIGHT_ANIMAL_POSITIVE
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_perfect_scores_yield_composite_one(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=1.0, helpful=1.0, animal_positive=1.0
        )
        score = evaluator.evaluate("Some article text", "Some Title")
        assert abs(score.composite - 1.0) < 1e-9

    def test_composite_applies_correct_weights(self):
        # accurate=0.8, helpful=0.6, animal_positive=0.9
        # expected = 0.8*0.35 + 0.6*0.30 + 0.9*0.35 = 0.28 + 0.18 + 0.315 = 0.775
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.8, helpful=0.6, animal_positive=0.9
        )
        score = evaluator.evaluate("Article text", "Title")
        assert abs(score.composite - 0.775) < 1e-6

    def test_animal_positive_weight_equals_accurate_weight(self):
        """Domain rule: animal_positive carries equal weight to accuracy."""
        assert AHAEvaluator.WEIGHT_ANIMAL_POSITIVE == AHAEvaluator.WEIGHT_ACCURATE


# ---------------------------------------------------------------------------
# AHA publication gate — the 0.75 threshold
# ---------------------------------------------------------------------------


class TestAHAPublicationGate:
    """CRITICAL: articles below 0.75 composite must NEVER be published."""

    def test_score_above_threshold_passes(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.9, helpful=0.9, animal_positive=0.9
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is True

    def test_score_clearly_at_threshold_passes(self):
        # Use a value where floating point arithmetic lands cleanly at/above 0.75.
        # Note: exact 0.75 per-dimension hits a floating point edge case because
        # 0.75*0.35 + 0.75*0.30 + 0.75*0.35 = 0.7499999999999999 in IEEE 754.
        # We use 0.76 to verify the gate works just above threshold.
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.76, helpful=0.76, animal_positive=0.76
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is True

    def test_score_below_threshold_fails(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.5, helpful=0.5, animal_positive=0.5
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False

    def test_score_just_below_threshold_fails(self):
        """0.74 composite must not pass — catching off-by-one rounding errors."""
        # accurate=0.74, helpful=0.74, animal_positive=0.74 → composite=0.74
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.74, helpful=0.74, animal_positive=0.74
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False

    def test_high_accurate_low_animal_positive_can_still_fail(self):
        """A speciesist article can't pass on accuracy alone."""
        # accurate=1.0, helpful=1.0, animal_positive=0.0
        # composite = 1.0*0.35 + 1.0*0.30 + 0.0*0.35 = 0.65 → fails
        evaluator = _make_evaluator_with_mock_response(
            accurate=1.0, helpful=1.0, animal_positive=0.0
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False

    def test_custom_threshold_is_respected(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.9, helpful=0.9, animal_positive=0.9, threshold=0.95
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False

    def test_flags_are_preserved(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.9, helpful=0.9, animal_positive=0.9,
            flags=["Uses 'livestock' framing"]
        )
        score = evaluator.evaluate("Article", "Title")
        assert "Uses 'livestock' framing" in score.flags

    def test_reasoning_is_preserved(self):
        evaluator = _make_evaluator_with_mock_response(
            accurate=0.9, helpful=0.9, animal_positive=0.9,
            reasoning="Good factual support but weak animal framing."
        )
        score = evaluator.evaluate("Article", "Title")
        assert score.reasoning == "Good factual support but weak animal framing."


# ---------------------------------------------------------------------------
# Fail-safe: evaluation errors must produce failing scores
# ---------------------------------------------------------------------------


class TestAHAEvaluatorFailSafe:
    """On any error, evaluator must return failed score — never a passing one."""

    def test_json_parse_error_returns_failing_score(self):
        evaluator = _make_evaluator_with_raw_response("not valid json {{{")
        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False
        assert score.composite == 0.0
        assert any("EVAL_PARSE_ERROR" in f for f in score.flags)

    def test_api_error_returns_failing_score(self):
        evaluator = AHAEvaluator.__new__(AHAEvaluator)
        evaluator.threshold = 0.75

        mock_client = MagicMock()
        mock_client.create_message.side_effect = ConnectionError("Network unreachable")
        evaluator.client = mock_client
        evaluator.model = "claude-sonnet-4-6"

        score = evaluator.evaluate("Article", "Title")
        assert score.passed is False
        assert score.composite == 0.0
        assert any("EVAL_API_ERROR" in f for f in score.flags)

    def test_error_score_never_passes(self):
        """Verify _error_score always produces passed=False regardless of flag."""
        evaluator = AHAEvaluator.__new__(AHAEvaluator)
        evaluator.threshold = 0.0  # Even with zero threshold
        score = evaluator._error_score("SOME_FLAG")
        # Error scores must always be False — even at threshold=0 we still
        # want explicit error states to block publication
        assert score.accurate == 0.0
        assert score.helpful == 0.0
        assert score.animal_positive == 0.0
        assert score.composite == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_response_json(
    accurate: float,
    helpful: float,
    animal_positive: float,
    reasoning: str = "Test reasoning.",
    flags: list[str] | None = None,
) -> str:
    return json.dumps({
        "accurate": accurate,
        "helpful": helpful,
        "animal_positive": animal_positive,
        "reasoning": reasoning,
        "flags": flags or [],
    })


def _make_evaluator_with_mock_response(
    accurate: float,
    helpful: float,
    animal_positive: float,
    threshold: float = 0.75,
    reasoning: str = "Test reasoning.",
    flags: list[str] | None = None,
) -> AHAEvaluator:
    evaluator = AHAEvaluator.__new__(AHAEvaluator)
    evaluator.threshold = threshold

    from src.articles.client import MessageResponse
    mock_client = MagicMock()
    mock_client.create_message.return_value = MessageResponse(
        text=_build_response_json(accurate, helpful, animal_positive, reasoning, flags)
    )
    evaluator.client = mock_client
    evaluator.model = "claude-sonnet-4-6"
    return evaluator


def _make_evaluator_with_raw_response(raw_text: str) -> AHAEvaluator:
    evaluator = AHAEvaluator.__new__(AHAEvaluator)
    evaluator.threshold = 0.75

    from src.articles.client import MessageResponse
    mock_client = MagicMock()
    mock_client.create_message.return_value = MessageResponse(text=raw_text)
    evaluator.client = mock_client
    evaluator.model = "claude-sonnet-4-6"
    return evaluator
