"""Speech-to-text transcription using the Groq Whisper API."""

from __future__ import annotations

import logging
from typing import Any

from groq import Groq

from core.config import GROQ_API_KEY, GROQ_WHISPER_MODEL

logger = logging.getLogger(__name__)


def _extract_language(response: Any) -> str:
    """Extract the detected language from a Groq verbose_json response object."""
    language = getattr(response, "language", None)
    # Groq's Transcription Pydantic model only statically declares `text`;
    # verbose_json fields like `language` may live in model_extra / model_dump().
    if not language and hasattr(response, "model_dump"):
        language = response.model_dump().get("language")
    if isinstance(language, str) and language.strip():
        return language.strip().lower()
    return "en"


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> dict[str, str]:
    """Transcribe audio bytes via Groq Whisper with automatic language detection."""
    if not audio_bytes:
        logger.warning("Skipping STT transcription because audio_bytes is empty.")
        return {"text": "", "language": "en"}

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing; cannot call Groq Whisper API.")
        return {"text": "", "language": "en"}

    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Omit `language` so Whisper auto-detects English, Hindi, Punjabi, etc.
        response = client.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=(filename, audio_bytes),
            response_format="verbose_json",
        )
        text = (getattr(response, "text", "") or "").strip()
        language = _extract_language(response)
        return {"text": text, "language": language}
    except Exception:
        logger.exception("Groq Whisper transcription failed.")
        return {"text": "", "language": "en"}
