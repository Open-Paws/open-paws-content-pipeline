"""NAV Language Checker client.

Calls mcp-server-nav-language to detect speciesist language violations in
generated content before it enters the AHA gate or HuggingFace pipeline.

Zero dependencies beyond stdlib + httpx (already a dependency for other services).

Design decisions:
- Synchronous: matches the pipeline's synchronous architecture throughout.
- Fail-open on service error: the AHA gate remains the primary quality control.
  A NAV outage should warn, not block the pipeline at scale.
- ERROR severity → is_clean=False (pipeline should block the article).
- WARNING severity → logged but not blocking (flagged for review).
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5
_MAX_CONTENT_CHARS = 32_000


def _nav_url() -> str:
    return os.environ.get("NAV_SERVER_URL", "http://localhost:3001").rstrip("/")


def check_article_language(
    content: str,
    article_id: str | None = None,
) -> dict[str, Any]:
    """Check article text for speciesist language violations.

    Args:
        content: Article text to check (truncated at 32K chars).
        article_id: Optional article ID for log correlation.

    Returns:
        Dict with keys:
          - is_clean: bool — False if any ERROR-severity violations found.
          - error_count: int — number of ERROR-severity violations.
          - warning_count: int — number of WARNING-severity violations.
          - violations: list of violation dicts (rule_id, severity, matched_text, message).
          - service_error: str or None — set if NAV server is unreachable.
    """
    truncated = content[:_MAX_CONTENT_CHARS]
    url = f"{_nav_url()}/check"
    log_prefix = f"[nav] article_id={article_id or 'unknown'}"

    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json={"text": truncated, "file_type": "prose"})
            if not resp.is_success:
                logger.warning(
                    "%s NAV server returned HTTP %s — treating as service error",
                    log_prefix,
                    resp.status_code,
                )
                return _service_error_result(f"HTTP {resp.status_code}")

            try:
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError(f"unexpected NAV response type: {type(data).__name__}")
                violations = data.get("violations", [])
            except Exception as parse_exc:
                logger.warning(
                    "%s NAV response parse error — treating as service error: %s",
                    log_prefix,
                    parse_exc,
                )
                return _service_error_result(f"parse error: {parse_exc}")

            errors = [v for v in violations if v.get("severity") == "error"]
            warnings = [v for v in violations if v.get("severity") == "warning"]

            logger.info(
                "%s errors=%d warnings=%d",
                log_prefix,
                len(errors),
                len(warnings),
            )
            return {
                "is_clean": len(errors) == 0,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "violations": violations,
                "service_error": None,
            }

    except httpx.ConnectError:
        logger.warning("%s NAV server unreachable at %s", log_prefix, _nav_url())
        return _service_error_result("NAV server unreachable")
    except httpx.TimeoutException:
        logger.warning(
            "%s NAV server timed out after %ss", log_prefix, _DEFAULT_TIMEOUT_SECONDS
        )
        return _service_error_result("NAV server timeout")
    except httpx.RequestError as exc:
        logger.warning("%s NAV request error: %s", log_prefix, exc)
        return _service_error_result(str(exc))


def _service_error_result(error: str) -> dict[str, Any]:
    """Return a non-blocking result when the NAV service is unavailable.

    NAV service errors are fail-open by design — the AHA gate is the primary
    quality control. A NAV outage should warn, not block the pipeline.
    """
    return {
        "is_clean": True,  # fail-open when service is down (AHA gate still runs)
        "error_count": 0,
        "warning_count": 0,
        "violations": [],
        "service_error": error,
    }
