"""
Multilingual content pipeline.

Translates English articles into target languages using claude-haiku.
Adapted from Synthalingua patterns (multi-provider, language-aware prompting).

Target languages: Hindi (hi), Spanish (es), French (fr), Mandarin (zh)

Key constraint: movement terminology must survive translation.
The system prompt explicitly instructs the model to preserve:
- Species names as-is or with verified local equivalents
- Movement terms with parenthetical English where no clean equivalent exists

Translation quality: claude-haiku produces serviceable translations for
training data purposes. High-stakes public-facing content should go through
a native speaker review queue before publication.
"""

import os
from dataclasses import dataclass
from typing import Optional

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "zh": "Mandarin Chinese",
}

_TRANSLATION_SYSTEM = """You are translating animal advocacy content for the Open Paws platform.

Rules:
1. Preserve factual accuracy completely — do not soften or alter claims
2. Use natural, fluent {target_language} appropriate for a general audience
3. Keep movement terminology precise:
   - "farmed animals" → translate accurately, not with industry euphemisms
   - "factory farm" → use the most accurate direct equivalent, not euphemisms
   - "slaughterhouse" → use the precise term, not euphemisms
4. Preserve any specific numbers, statistics, or proper nouns
5. Do not add content or remove content — translate what is there

Output only the translated text, no preamble."""


@dataclass
class TranslatedArticle:
    title: str
    body: str
    language: str
    source_language: str = "en"
    source_title: str = ""
    source_body: str = ""
    review_needed: bool = True  # All machine translations need human review for high-stakes use


class Translator:
    """
    Translate articles to target languages.

    review_needed is always True on output — machine translations for
    training data are acceptable, but public-facing use requires human review.
    """

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("TRANSLATION_MODEL", "claude-haiku-4-5-20251001")

    def translate(
        self,
        title: str,
        body: str,
        target_language: str,
    ) -> Optional[TranslatedArticle]:
        """
        Translate a single article to target_language.

        target_language: ISO 639-1 code ("hi", "es", "fr", "zh")
        Returns None if language is unsupported or API call fails.
        """
        if target_language not in SUPPORTED_LANGUAGES:
            return None

        lang_name = SUPPORTED_LANGUAGES[target_language]
        system = _TRANSLATION_SYSTEM.format(target_language=lang_name)

        try:
            # Translate title and body together to maintain context
            content = f"TITLE: {title}\n\nBODY:\n{body}"
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
            translated = response.content[0].text.strip()

            # Split on the body separator
            if "BODY:" in translated:
                parts = translated.split("BODY:", 1)
                translated_title = parts[0].replace("TITLE:", "").strip()
                translated_body = parts[1].strip()
            else:
                # Fallback: treat full response as body, keep original title
                translated_title = title
                translated_body = translated

            return TranslatedArticle(
                title=translated_title,
                body=translated_body,
                language=target_language,
                source_language="en",
                source_title=title,
                source_body=body,
                review_needed=True,
            )
        except Exception:
            return None

    def translate_to_all(
        self,
        title: str,
        body: str,
        languages: Optional[list[str]] = None,
    ) -> list[TranslatedArticle]:
        """
        Translate to multiple languages. Returns only successful translations.

        Default: all four supported languages (hi, es, fr, zh).
        """
        targets = languages or list(SUPPORTED_LANGUAGES.keys())
        results = []
        for lang in targets:
            translation = self.translate(title, body, lang)
            if translation is not None:
                results.append(translation)
        return results
