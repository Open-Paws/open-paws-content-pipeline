"""
AI client factory for the content pipeline.

Priority order for API backend:
  1. ANTHROPIC_API_KEY → use Anthropic SDK directly (preferred, native support)
  2. OPENROUTER_API_KEY → use OpenRouter via OpenAI-compatible API

Both backends expose the same interface so callers are not coupled to a provider.
OpenRouter model names use "anthropic/" prefix (e.g. "anthropic/claude-haiku-4-5").

Model name translation:
  claude-haiku-4-5-20251001 → anthropic/claude-haiku-4-5   (OpenRouter)
  claude-sonnet-4-6         → anthropic/claude-sonnet-4-6  (OpenRouter)
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

# Translate Anthropic model IDs to OpenRouter model IDs
_OPENROUTER_MODEL_MAP: dict[str, str] = {
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "claude-opus-4-5": "anthropic/claude-opus-4-5",
}


@dataclass
class MessageResponse:
    """Unified response object compatible with both Anthropic and OpenAI SDK responses."""
    text: str


class _AnthropicClient:
    """Thin wrapper around the Anthropic SDK."""

    def __init__(self) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        self.backend = "anthropic"

    def create_message(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> MessageResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return MessageResponse(text=response.content[0].text.strip())


class _OpenRouterClient:
    """OpenAI-compatible client pointing at OpenRouter, routing to Claude models."""

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.backend = "openrouter"

    def _translate_model(self, model: str) -> str:
        """Convert Anthropic model ID to OpenRouter model ID."""
        return _OPENROUTER_MODEL_MAP.get(model, f"anthropic/{model}")

    def create_message(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> MessageResponse:
        or_model = self._translate_model(model)
        or_messages = []
        if system:
            or_messages.append({"role": "system", "content": system})
        or_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=or_model,
            max_tokens=max_tokens,
            messages=or_messages,
            temperature=temperature,
        )
        return MessageResponse(text=response.choices[0].message.content.strip())


def get_client() -> "_AnthropicClient | _OpenRouterClient":
    """
    Return the best available AI client.

    Prefers Anthropic SDK when ANTHROPIC_API_KEY is set.
    Falls back to OpenRouter when OPENROUTER_API_KEY is set.
    Raises RuntimeError if neither key is available.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if anthropic_key:
        return _AnthropicClient()

    if openrouter_key:
        return _OpenRouterClient(api_key=openrouter_key)

    raise RuntimeError(
        "No AI API key configured. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY."
    )
