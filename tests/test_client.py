"""
Tests for the AI client factory.

Domain rules encoded:
- Anthropic key takes priority over OpenRouter key
- Missing both keys raises RuntimeError (not a silent failure)
- OpenRouter model name translation is correct for all configured models
- MessageResponse exposes .text for callers
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.articles.client import MessageResponse, _OPENROUTER_MODEL_MAP


# ---------------------------------------------------------------------------
# MessageResponse
# ---------------------------------------------------------------------------


class TestMessageResponse:
    def test_text_attribute_accessible(self):
        resp = MessageResponse(text="hello world")
        assert resp.text == "hello world"

    def test_empty_text_allowed(self):
        resp = MessageResponse(text="")
        assert resp.text == ""


# ---------------------------------------------------------------------------
# get_client — key priority
# ---------------------------------------------------------------------------


class TestGetClientPriority:
    def test_raises_when_no_key_set(self):
        """Both keys absent must raise RuntimeError — never silently fail."""
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            from src.articles import client as client_mod
            with pytest.raises(RuntimeError, match="No AI API key"):
                client_mod.get_client()

    def test_anthropic_key_returns_anthropic_backend(self):
        """When ANTHROPIC_API_KEY is set, backend must be 'anthropic'."""
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        # _AnthropicClient imports anthropic inside __init__; patch the module
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                from importlib import reload
                from src.articles import client as client_mod
                reload(client_mod)
                c = client_mod.get_client()
        assert c.backend == "anthropic"

    def test_openrouter_key_returns_openrouter_backend(self):
        """When only OPENROUTER_API_KEY is set, backend must be 'openrouter'."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k != "ANTHROPIC_API_KEY"
        }
        clean_env["OPENROUTER_API_KEY"] = "sk-or-test"
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from importlib import reload
                from src.articles import client as client_mod
                reload(client_mod)
                c = client_mod.get_client()
        assert c.backend == "openrouter"

    def test_anthropic_key_takes_priority_over_openrouter(self):
        """When both keys are set, Anthropic must win (cost + reliability)."""
        env = {
            "ANTHROPIC_API_KEY": "sk-anthropic",
            "OPENROUTER_API_KEY": "sk-openrouter",
        }
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                from importlib import reload
                from src.articles import client as client_mod
                reload(client_mod)
                c = client_mod.get_client()
        assert c.backend == "anthropic"


# ---------------------------------------------------------------------------
# OpenRouter model name translation
# ---------------------------------------------------------------------------


class TestOpenRouterModelTranslation:
    """Test _OpenRouterClient._translate_model() directly — no API call needed."""

    def _make_openrouter_client(self):
        from src.articles.client import _OpenRouterClient
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            # Instantiate without a real OpenAI client
            client = _OpenRouterClient.__new__(_OpenRouterClient)
            client.client = MagicMock()
            client.backend = "openrouter"
        return client

    def test_haiku_translates_correctly(self):
        """claude-haiku is the article generation model (cheap)."""
        c = self._make_openrouter_client()
        assert (
            c._translate_model("claude-haiku-4-5-20251001")
            == "anthropic/claude-haiku-4-5"
        )

    def test_sonnet_translates_correctly(self):
        """claude-sonnet-4-6 is the AHA evaluation model (judgment)."""
        c = self._make_openrouter_client()
        assert (
            c._translate_model("claude-sonnet-4-6")
            == "anthropic/claude-sonnet-4-6"
        )

    def test_unknown_model_gets_anthropic_prefix(self):
        """Unmapped models fall back to anthropic/<model> prefix."""
        c = self._make_openrouter_client()
        result = c._translate_model("claude-future-model")
        assert result == "anthropic/claude-future-model"

    def test_model_map_covers_generation_and_evaluation_models(self):
        """Both production models must be in the translation map."""
        assert "claude-haiku-4-5-20251001" in _OPENROUTER_MODEL_MAP
        assert "claude-sonnet-4-6" in _OPENROUTER_MODEL_MAP
