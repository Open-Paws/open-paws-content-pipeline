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
"""

import json
import os
from dataclasses import dataclass, field


EVALUATION_PROMPT = """You are evaluating an article for the Open Paws animal advocacy platform.
Score this article on three dimensions (0.0 to 1.0 each):

1. ACCURATE: Are all factual claims about animal welfare, agriculture, and science accurate?
   Deduct for: unsupported statistics, exaggerated claims, misrepresented research,
   fabricated numbers, or assertions without evidence.

2. HELPFUL: Is this useful for advocates or informative for a general audience?
   Score for: actionable information, clear explanations, relevant context,
   specific data points, and logical argument structure.

3. ANIMAL_POSITIVE: Does this avoid normalizing animal exploitation?
   Deduct for: industry framing ("livestock", "processing facility", "harvest"),
   euphemisms for killing ("culling", "dispatching"), or language that treats
   animals as commodities rather than sentient individuals.
   High score = uses movement terminology consistently (farmed animals, factory farm,
   slaughterhouse), centers animal experience, avoids speciesist idioms.

Respond with JSON only, no markdown:
{
  "accurate": float,
  "helpful": float,
  "animal_positive": float,
  "reasoning": "one paragraph explaining scores",
  "flags": ["specific issue 1", "specific issue 2"]
}"""


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
    Evaluate article quality before publication.

    Uses a configurable threshold (default 0.75). Articles below threshold
    go to review queue; they never auto-publish. Fail-safe: parsing errors
    return a failing score rather than a passing one.
    """

    # Weights must sum to 1.0
    WEIGHT_ACCURATE = 0.35
    WEIGHT_HELPFUL = 0.30
    WEIGHT_ANIMAL_POSITIVE = 0.35

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        # Lazy import to allow module load without API key present
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("AHA_EVAL_MODEL", "claude-sonnet-4-6")

    def evaluate(self, article_text: str, title: str) -> AHAScore:
        """
        Evaluate one article. Returns AHAScore with passed=True if above threshold.

        On any error (network, parsing), returns a failing score with the
        EVAL_ERROR flag set. Callers must check score.passed before publishing.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=EVALUATION_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Title: {title}\n\nArticle:\n{article_text}",
                }],
            )
            raw = response.content[0].text.strip()
            result = json.loads(raw)
        except json.JSONDecodeError:
            return self._error_score("EVAL_PARSE_ERROR")
        except Exception as exc:
            return self._error_score(f"EVAL_API_ERROR: {type(exc).__name__}")

        accurate = float(result.get("accurate", 0))
        helpful = float(result.get("helpful", 0))
        animal_positive = float(result.get("animal_positive", 0))

        composite = (
            accurate * self.WEIGHT_ACCURATE
            + helpful * self.WEIGHT_HELPFUL
            + animal_positive * self.WEIGHT_ANIMAL_POSITIVE
        )

        return AHAScore(
            accurate=accurate,
            helpful=helpful,
            animal_positive=animal_positive,
            composite=composite,
            reasoning=result.get("reasoning", ""),
            passed=composite >= self.threshold,
            flags=result.get("flags", []),
        )

    def _error_score(self, flag: str) -> AHAScore:
        return AHAScore(
            accurate=0.0,
            helpful=0.0,
            animal_positive=0.0,
            composite=0.0,
            reasoning="Evaluation failed.",
            passed=False,
            flags=[flag],
        )
