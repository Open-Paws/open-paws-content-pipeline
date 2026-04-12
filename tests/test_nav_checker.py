"""Tests for src/articles/nav_checker.py.

Each test must fail if the covered behaviour is broken — no dead assertions.

Test coverage:
  - clean content → is_clean=True, no violations
  - ERROR violations → is_clean=False, error_count > 0
  - WARNING-only violations → is_clean=True (warnings do not block)
  - NAV server down (ConnectError) → fail-open: is_clean=True, service_error set
  - NAV server timeout → fail-open: is_clean=True, service_error set
  - HTTP non-success response → fail-open: is_clean=True, service_error set
  - Content truncation at 32K chars
"""

from unittest.mock import MagicMock, patch

import httpx

from src.articles.nav_checker import (
    _MAX_CONTENT_CHARS,
    _service_error_result,
    check_article_language,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(violations: list[dict], status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = (200 <= status_code < 300)
    resp.json.return_value = {"violations": violations}
    return resp


def _mock_client(response: MagicMock) -> MagicMock:
    """Build a fake httpx.Client context manager that returns a fixed response."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.post.return_value = response
    return client


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestCheckArticleCleanContent:
    """NAV server returns no violations → article is clean."""

    def test_is_clean_true(self):
        resp = _mock_response(violations=[])
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Farmed animals deserve legal protection.", article_id="a1")

        assert result["is_clean"] is True

    def test_error_count_zero(self):
        resp = _mock_response(violations=[])
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Farmed animals deserve legal protection.")

        assert result["error_count"] == 0

    def test_warning_count_zero(self):
        resp = _mock_response(violations=[])
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Farmed animals deserve legal protection.")

        assert result["warning_count"] == 0

    def test_service_error_is_none(self):
        resp = _mock_response(violations=[])
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Farmed animals deserve legal protection.")

        assert result["service_error"] is None

    def test_violations_empty(self):
        resp = _mock_response(violations=[])
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Farmed animals deserve legal protection.")

        assert result["violations"] == []


class TestCheckArticleWithErrors:
    """NAV server returns ERROR-severity violations → article is blocked."""

    _ERROR_VIOLATIONS = (
        {
            "rule_id": "no-livestock",
            "severity": "error",
            "matched_text": "livestock",
            "message": "Use 'farmed animals' instead of 'livestock'.",
        },
        {
            "rule_id": "no-processing-facility",
            "severity": "error",
            "matched_text": "processing facility",
            "message": "Use 'slaughterhouse' instead of 'processing facility'.",
        },
    )

    def test_is_clean_false(self):
        resp = _mock_response(violations=self._ERROR_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("livestock sent to processing facility")

        assert result["is_clean"] is False

    def test_error_count_matches(self):
        resp = _mock_response(violations=self._ERROR_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("livestock sent to processing facility")

        assert result["error_count"] == 2

    def test_violations_included(self):
        resp = _mock_response(violations=self._ERROR_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("livestock sent to processing facility")

        assert len(result["violations"]) == 2

    def test_service_error_still_none(self):
        """A successful NAV call with errors should not set service_error."""
        resp = _mock_response(violations=self._ERROR_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("livestock sent to processing facility")

        assert result["service_error"] is None


class TestCheckArticleWithWarningsOnly:
    """WARNING-only violations do not block — is_clean remains True."""

    _WARNING_VIOLATIONS = (
        {
            "rule_id": "prefer-factory-farm",
            "severity": "warning",
            "matched_text": "farm",
            "message": "Consider 'factory farm' for precision.",
        },
    )

    def test_is_clean_true_with_warnings(self):
        resp = _mock_response(violations=self._WARNING_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Animals suffer on the farm.")

        assert result["is_clean"] is True

    def test_warning_count_set(self):
        resp = _mock_response(violations=self._WARNING_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Animals suffer on the farm.")

        assert result["warning_count"] == 1

    def test_error_count_zero_with_warnings(self):
        resp = _mock_response(violations=self._WARNING_VIOLATIONS)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Animals suffer on the farm.")

        assert result["error_count"] == 0


# ---------------------------------------------------------------------------
# Fail-open (service unavailable) tests
# ---------------------------------------------------------------------------

class TestNavServerDownReturnsFailOpen:
    """ConnectError → fail-open: pipeline continues to AHA gate."""

    def test_is_clean_true_on_connect_error(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.ConnectError("refused")
            result = check_article_language("Some article text.", article_id="b1")

        assert result["is_clean"] is True

    def test_service_error_set_on_connect_error(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.ConnectError("refused")
            result = check_article_language("Some article text.")

        assert result["service_error"] is not None
        assert "unreachable" in result["service_error"].lower()

    def test_error_count_zero_on_connect_error(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.ConnectError("refused")
            result = check_article_language("Some article text.")

        assert result["error_count"] == 0

    def test_violations_empty_on_connect_error(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.ConnectError("refused")
            result = check_article_language("Some article text.")

        assert result["violations"] == []


class TestNavServerTimeoutReturnsFailOpen:
    """TimeoutException → fail-open: pipeline continues to AHA gate."""

    def test_is_clean_true_on_timeout(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.TimeoutException("timed out")
            result = check_article_language("Some article text.", article_id="c1")

        assert result["is_clean"] is True

    def test_service_error_set_on_timeout(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.TimeoutException("timed out")
            result = check_article_language("Some article text.")

        assert result["service_error"] is not None
        assert "timeout" in result["service_error"].lower()

    def test_error_count_zero_on_timeout(self):
        with patch("src.articles.nav_checker.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.side_effect = httpx.TimeoutException("timed out")
            result = check_article_language("Some article text.")

        assert result["error_count"] == 0


class TestNavServerNonSuccessResponse:
    """HTTP non-2xx response → fail-open: service_error set, is_clean=True."""

    def test_is_clean_true_on_http_error(self):
        resp = _mock_response(violations=[], status_code=503)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Some article text.")

        assert result["is_clean"] is True

    def test_service_error_includes_status_code(self):
        resp = _mock_response(violations=[], status_code=503)
        with patch("src.articles.nav_checker.httpx.Client", return_value=_mock_client(resp)):
            result = check_article_language("Some article text.")

        assert result["service_error"] is not None
        assert "503" in result["service_error"]


# ---------------------------------------------------------------------------
# Content truncation
# ---------------------------------------------------------------------------

class TestContentTruncation:
    """Content exceeding 32K chars is truncated before sending."""

    def test_long_content_is_truncated(self):
        long_content = "A" * (_MAX_CONTENT_CHARS + 10_000)
        resp = _mock_response(violations=[])
        client = _mock_client(resp)

        with patch("src.articles.nav_checker.httpx.Client", return_value=client):
            check_article_language(long_content)

        call_kwargs = client.post.call_args
        sent_text = call_kwargs.kwargs["json"]["text"] if call_kwargs.kwargs else call_kwargs[1]["json"]["text"]
        assert len(sent_text) == _MAX_CONTENT_CHARS

    def test_short_content_is_not_truncated(self):
        short_content = "Farmed animals deserve protection."
        resp = _mock_response(violations=[])
        client = _mock_client(resp)

        with patch("src.articles.nav_checker.httpx.Client", return_value=client):
            check_article_language(short_content)

        call_kwargs = client.post.call_args
        sent_text = call_kwargs.kwargs["json"]["text"] if call_kwargs.kwargs else call_kwargs[1]["json"]["text"]
        assert sent_text == short_content


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class TestServiceErrorResult:
    """_service_error_result always returns a fail-open dict."""

    def test_is_clean_true(self):
        result = _service_error_result("some error")
        assert result["is_clean"] is True

    def test_error_count_zero(self):
        result = _service_error_result("some error")
        assert result["error_count"] == 0

    def test_service_error_preserved(self):
        result = _service_error_result("NAV server unreachable")
        assert result["service_error"] == "NAV server unreachable"

    def test_violations_empty(self):
        result = _service_error_result("timeout")
        assert result["violations"] == []
