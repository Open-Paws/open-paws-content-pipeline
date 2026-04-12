"""
Tests for multilingual translation pipeline.

Domain rules encoded:
- Only supported ISO 639-1 codes are accepted (hi, es, fr, zh)
- Unsupported language codes must return None (not raise)
- review_needed is always True on machine-translated output
- TranslatedArticle carries source and target language tags
- translate_to_all returns only successful translations
"""

from unittest.mock import MagicMock


from src.multilingual.translator import SUPPORTED_LANGUAGES, TranslatedArticle, Translator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_translator(translated_text: str) -> Translator:
    """Build a Translator with mocked Anthropic client."""
    translator = Translator.__new__(Translator)
    translator.model = "claude-haiku-4-5-20251001"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=translated_text)]
    mock_client.messages.create.return_value = mock_response
    translator.client = mock_client

    return translator


def _make_failing_translator() -> Translator:
    """Build a Translator whose API call always raises."""
    translator = Translator.__new__(Translator)
    translator.model = "claude-haiku-4-5-20251001"

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ConnectionError("Network error")
    translator.client = mock_client

    return translator


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------


class TestSupportedLanguages:
    def test_supported_languages_includes_hindi(self):
        assert "hi" in SUPPORTED_LANGUAGES

    def test_supported_languages_includes_spanish(self):
        assert "es" in SUPPORTED_LANGUAGES

    def test_supported_languages_includes_french(self):
        assert "fr" in SUPPORTED_LANGUAGES

    def test_supported_languages_includes_mandarin(self):
        assert "zh" in SUPPORTED_LANGUAGES

    def test_unsupported_language_returns_none(self):
        translator = _make_translator("Some translation")
        result = translator.translate("Title", "Body", "xx")  # Invalid code
        assert result is None

    def test_english_not_in_supported_languages(self):
        """English is the source language; translating to 'en' makes no sense."""
        assert "en" not in SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# TranslatedArticle output structure
# ---------------------------------------------------------------------------


class TestTranslatedArticleStructure:
    def test_translate_returns_translated_article(self):
        translated_text = "TITLE: Translated Title\n\nBODY:\nTranslated body content."
        translator = _make_translator(translated_text)
        result = translator.translate("English Title", "English body.", "es")
        assert isinstance(result, TranslatedArticle)

    def test_language_tag_matches_target(self):
        translated_text = "TITLE: Hindi Title\n\nBODY:\nHindi body."
        translator = _make_translator(translated_text)
        result = translator.translate("English Title", "English body.", "hi")
        assert result is not None
        assert result.language == "hi"

    def test_source_language_is_english(self):
        translated_text = "TITLE: Spanish Title\n\nBODY:\nSpanish body."
        translator = _make_translator(translated_text)
        result = translator.translate("English Title", "English body.", "es")
        assert result is not None
        assert result.source_language == "en"

    def test_review_needed_is_always_true(self):
        """Machine translations always require human review before public use."""
        translated_text = "TITLE: French Title\n\nBODY:\nFrench body."
        translator = _make_translator(translated_text)
        result = translator.translate("English Title", "English body.", "fr")
        assert result is not None
        assert result.review_needed is True

    def test_source_title_is_preserved(self):
        translated_text = "TITLE: Translated\n\nBODY:\nBody."
        translator = _make_translator(translated_text)
        result = translator.translate("Original Title", "Body.", "zh")
        assert result is not None
        assert result.source_title == "Original Title"

    def test_source_body_is_preserved(self):
        translated_text = "TITLE: Translated\n\nBODY:\nTranslated body."
        translator = _make_translator(translated_text)
        result = translator.translate("Title", "Original body content.", "es")
        assert result is not None
        assert result.source_body == "Original body content."


# ---------------------------------------------------------------------------
# API failure handling
# ---------------------------------------------------------------------------


class TestTranslationFailureHandling:
    def test_api_error_returns_none(self):
        translator = _make_failing_translator()
        result = translator.translate("Title", "Body", "hi")
        assert result is None

    def test_none_result_does_not_raise(self):
        translator = _make_failing_translator()
        # Should not raise, just return None
        result = translator.translate("Title", "Body", "fr")
        assert result is None


# ---------------------------------------------------------------------------
# translate_to_all
# ---------------------------------------------------------------------------


class TestTranslateToAll:
    def test_translate_to_all_returns_list(self):
        translated_text = "TITLE: T\n\nBODY:\nB."
        translator = _make_translator(translated_text)
        results = translator.translate_to_all("Title", "Body")
        assert isinstance(results, list)

    def test_translate_to_all_default_covers_all_supported_languages(self):
        translated_text = "TITLE: T\n\nBODY:\nB."
        translator = _make_translator(translated_text)
        results = translator.translate_to_all("Title", "Body")
        returned_langs = {r.language for r in results}
        assert returned_langs == set(SUPPORTED_LANGUAGES.keys())

    def test_translate_to_all_with_subset_of_languages(self):
        translated_text = "TITLE: T\n\nBODY:\nB."
        translator = _make_translator(translated_text)
        results = translator.translate_to_all("Title", "Body", languages=["es", "fr"])
        returned_langs = {r.language for r in results}
        assert returned_langs == {"es", "fr"}

    def test_translate_to_all_skips_failed_translations(self):
        """Partial failure must not block the successful translations."""
        translator = Translator.__new__(Translator)
        translator.model = "claude-haiku-4-5-20251001"

        mock_client = MagicMock()
        # First call succeeds, second fails
        success_response = MagicMock()
        success_response.content = [MagicMock(text="TITLE: T\n\nBODY:\nB.")]
        mock_client.messages.create.side_effect = [
            success_response,
            ConnectionError("fail"),
        ]
        translator.client = mock_client

        results = translator.translate_to_all("Title", "Body", languages=["es", "hi"])
        # Only one success
        assert len(results) == 1
