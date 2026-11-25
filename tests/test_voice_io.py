"""Unit tests for Stage 2 voice I/O modules (core/stt.py and core/tts.py)."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from core import stt, tts


class TestVoiceIO(unittest.TestCase):
    """Verify STT transcription, TTS synthesis, language mapping, and error fallbacks."""

    def test_stt_empty_audio_returns_fallback(self) -> None:
        """Empty audio input should not invoke Groq and should return an empty transcript."""
        result = stt.transcribe(b"")
        self.assertEqual(result, {"text": "", "language": "en"})

    def test_stt_transcribe_success_auto_detects_language(self) -> None:
        """Verify transcribe omits forced language and extracts detected language."""
        mock_response = SimpleNamespace(text="  Namaste, mujhe demo chahiye  ", language="Hindi")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        with patch.object(stt, "GROQ_API_KEY", "test-key"), patch.object(
            stt, "Groq", return_value=mock_client
        ):
            result = stt.transcribe(b"fake-wav-bytes")

        self.assertEqual(
            result,
            {"text": "Namaste, mujhe demo chahiye", "language": "hindi"},
        )
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(call_kwargs["response_format"], "verbose_json")
        self.assertNotIn("language", call_kwargs)

    def test_stt_api_error_returns_graceful_fallback(self) -> None:
        """Verify Groq API exceptions are caught and return the safe fallback dict."""
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("API down")

        with patch.object(stt, "GROQ_API_KEY", "test-key"), patch.object(
            stt, "Groq", return_value=mock_client
        ):
            result = stt.transcribe(b"fake-wav-bytes")

        self.assertEqual(result, {"text": "", "language": "en"})

    def test_tts_language_mapping_and_fallback(self) -> None:
        """Verify Whisper language names/codes map to gTTS codes with 'en' fallback."""
        self.assertEqual(tts.normalize_lang_code("English"), "en")
        self.assertEqual(tts.normalize_lang_code("hindi"), "hi")
        self.assertEqual(tts.normalize_lang_code("Punjabi"), "pa")
        self.assertEqual(tts.normalize_lang_code("pa"), "pa")
        self.assertEqual(tts.normalize_lang_code("unknown-lang"), "en")
        self.assertEqual(tts.normalize_lang_code(None), "en")

    def test_tts_speak_empty_and_success(self) -> None:
        """Verify speak handles empty text, writes MP3 bytes to memory, and handles errors."""
        self.assertEqual(tts.speak("   ", "en"), b"")

        fake_tts_instance = MagicMock()
        fake_tts_instance.write_to_fp.side_effect = lambda fp: fp.write(b"ID3-MP3-DATA")

        with patch.object(tts, "gTTS", return_value=fake_tts_instance) as mock_gtts:
            audio = tts.speak("Sat Sri Akal", "punjabi")

        mock_gtts.assert_called_once_with(text="Sat Sri Akal", lang="pa")
        self.assertEqual(audio, b"ID3-MP3-DATA")

        with patch.object(tts, "gTTS", side_effect=RuntimeError("Network error")):
            self.assertEqual(tts.speak("Hello", "en"), b"")


if __name__ == "__main__":
    unittest.main()

