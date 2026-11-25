"""Text-to-speech synthesis using gTTS with in-memory MP3 output."""

from __future__ import annotations

from io import BytesIO
import logging

from gtts import gTTS

logger = logging.getLogger(__name__)

# Map Whisper verbose_json language names and ISO codes to gTTS language codes.
LANGUAGE_TO_GTTS: dict[str, str] = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "pa": "pa",
    "punjabi": "pa",
    "panjabi": "pa",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
}


def normalize_lang_code(lang_code: str | None) -> str:
    """Map a Whisper language name or code to a supported gTTS code, defaulting to 'en'."""
    if not lang_code:
        return "en"
    cleaned = lang_code.strip().lower()
    return LANGUAGE_TO_GTTS.get(cleaned, "en")


def speak(text: str, lang_code: str = "en") -> bytes:
    """Synthesize spoken MP3 audio bytes in memory using gTTS."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        logger.warning("Skipping TTS synthesis because text is empty.")
        return b""

    target_lang = normalize_lang_code(lang_code)
    try:
        audio_buffer = BytesIO()
        tts = gTTS(text=cleaned_text, lang=target_lang)
        tts.write_to_fp(audio_buffer)
        return audio_buffer.getvalue()
    except Exception:
        logger.exception("gTTS synthesis failed for lang='%s'.", target_lang)
        return b""
