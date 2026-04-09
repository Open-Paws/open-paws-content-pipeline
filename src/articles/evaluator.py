"""
AHA (Accurate, Helpful, Animal-positive) evaluation gate.

Every article must pass before publication.
No article auto-publishes below the configured threshold.

Scoring dimensions:
- Accurate (35%): factually correct claims about animal welfare and agriculture
- Helpful (30%): actionable for advocates or informative for general audience
- Animal-positive (35%): avoids industry framing, does not normalize exploitation

The animal_positive dimension carries equal weight to accuracy because
speciesist framing in published content undermines the training data mission.

Evaluation is performed by the shared AHA MCP server rather than a direct
Claude prompt, so all cost tracking and key management is centralised.
"""

import os
from dataclasses import dataclass, field

import httpx


@dataclass
class AHAScore:
    accurate: float
    helpful: float
    animal_positive: float
    composite: float
    reasoning: str
    passed: bool
    flags: list[str] = field(default_factory=list)


class AHAEvaluator:
    """
    Evaluate article quality before publication via the AHA MCP server.

    Uses a configurable threshold (default 0.75). Articles below threshold
    go to review queue; they never auto-publish. Fail-safe: server errors
    return a failing score rather than a passing one.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.server_url = os.environ.get("AHA_MCP_SERVER_URL", "http://localhost:3001").rstrip("/")

    def evaluate(self, article_text: str, title: str) -> AHAScore:
        """
        Evaluate one article. Returns AHAScore with passed=True if above threshold.

        On any error (network, parsing), returns a failing score with the
        AHA_ERROR flag set. Callers must check score.passed before publishing.
        """
        payload = {
            "content": f"Title: {title}\n\n{article_text}",
            "context": "training_data",
            "threshold": self.threshold,
        }
        try:
            resp = httpx.post(
                f"{self.server_url}/tools/evaluate_animal_harm",
                json=payload,
                headers={"X-Data-Collection": "deny"},
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            return AHAScore(0.0, 0.0, 0.0, 0.0, "Evaluation failed.", False, [f"AHA_ERROR: {exc}"])

        dims = result.get("dimension_scores", {})
        score = float(result.get("score", 0.0))
        flags = [f.get("message", "") for f in result.get("flags", [])]
        return AHAScore(
            accurate=float(dims.get("factual_accuracy", 0.0)),
            helpful=float(dims.get("welfare_framing", 0.0)),
            animal_positive=float(dims.get("speciesist_language", 0.0)),
            composite=score,
            reasoning=result.get("model_used", "mcp-server"),
            passed=result.get("recommendation") != "reject" and score >= self.threshold,
            flags=flags,
        )
