"""Unit tests for Stage 3 conversational brain (core/llm.py and core/memory.py)."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from core import db, llm, memory


class TestConversationalBrain(unittest.TestCase):
    """Verify JSON parsing, LLM turn generation, rolling memory, and SQLite persistence."""

    def setUp(self) -> None:
        """Create an isolated temporary SQLite database for memory persistence tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_brain.db"
        db.init_db(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database files."""
        self.temp_dir.cleanup()

    def test_parse_turn_json_strips_fences_and_clamps_values(self) -> None:
        """Verify fenced JSON is parsed cleanly and sentiment_score is clamped to [-1, 1]."""
        raw = """```json
        {
          "reply": "Hello! What is your name?",
          "language": "EN",
          "sentiment": "POSITIVE",
          "sentiment_score": 2.5,
          "intent": "pricing",
          "extracted": {"name": "Asha", "budget": "$500/mo"},
          "wants_booking": true,
          "preferred_time": "Tomorrow 3 PM"
        }
        ```"""
        parsed = llm._parse_turn_json(raw, default_lang="en")
        self.assertEqual(parsed["reply"], "Hello! What is your name?")
        self.assertEqual(parsed["language"], "en")
        self.assertEqual(parsed["sentiment"], "positive")
        self.assertEqual(parsed["sentiment_score"], 1.0)
        self.assertEqual(parsed["intent"], "pricing")
        self.assertEqual(parsed["extracted"]["name"], "Asha")
        self.assertEqual(parsed["extracted"]["budget"], "$500/mo")
        self.assertIsNone(parsed["extracted"]["email"])
        self.assertTrue(parsed["wants_booking"])
        self.assertEqual(parsed["preferred_time"], "Tomorrow 3 PM")

    def test_parse_turn_json_falls_back_to_plain_reply(self) -> None:
        """Verify malformed JSON gracefully falls back to a plain-text reply."""
        parsed = llm._parse_turn_json("Sure, I can connect you with our team.", default_lang="hi")
        self.assertEqual(parsed["reply"], "Sure, I can connect you with our team.")
        self.assertEqual(parsed["language"], "hi")
        self.assertEqual(parsed["sentiment"], "neutral")
        self.assertFalse(parsed["wants_booking"])

    def test_generate_turn_calls_gemini_and_handles_errors(self) -> None:
        """Verify generate_turn passes history/context and handles API exceptions."""
        mock_client = MagicMock()
        mock_client.invoke.return_value = SimpleNamespace(
            content='{"reply": "Happy to help with pricing.", "language": "en", '
            '"sentiment": "positive", "sentiment_score": 0.7, "intent": "pricing", '
            '"extracted": {"name": "Raj"}, "wants_booking": false, "preferred_time": null}'
        )

        with patch.object(llm, "GEMINI_API_KEY", "test-key"), patch.object(
            llm, "_create_chat_client", return_value=mock_client
        ):
            result = llm.generate_turn(
                user_text="How much does it cost?",
                history=[{"role": "assistant", "text": "Hi there!"}],
                rag_context="Starter plan is $49/month.",
                lead_state={"name": None},
                detected_lang="en",
            )

        self.assertEqual(result["reply"], "Happy to help with pricing.")
        self.assertEqual(result["extracted"]["name"], "Raj")
        mock_client.invoke.assert_called_once()

        mock_client.invoke.side_effect = RuntimeError("Gemini timeout")
        with patch.object(llm, "GEMINI_API_KEY", "test-key"), patch.object(
            llm, "_create_chat_client", return_value=mock_client
        ):
            fallback = llm.generate_turn("Hello", [], "", {}, "en")
        self.assertIn("human", fallback["reply"].lower())

    def test_memory_rolls_window_and_persists_all_messages(self) -> None:
        """Verify messages > N=10 trigger summarization while persisting every turn to SQLite."""
        state: dict = {}
        call_id = memory.init_conversation_memory(state, db_path=self.db_path)

        with patch.object(
            memory,
            "summarize_messages",
            side_effect=lambda msgs, existing_summary="": f"Summary({len(msgs)} older)",
        ) as mock_summarize:
            for idx in range(12):
                role = "user" if idx % 2 == 0 else "assistant"
                memory.record_message(
                    state,
                    role=role,
                    text=f"Message {idx}",
                    language="en",
                    sentiment_label="positive" if role == "user" else None,
                    sentiment_score=0.5 if role == "user" else None,
                    max_window=10,
                    db_path=self.db_path,
                )

        # Window keeps latest 10 for the LLM, while full transcript + SQLite keep all 12.
        self.assertEqual(len(state["llm_history"]), 10)
        self.assertEqual(state["llm_history"][0]["text"], "Message 2")
        self.assertEqual(len(state["transcript"]), 12)
        self.assertEqual(mock_summarize.call_count, 2)
        self.assertTrue(state["memory_summary"].startswith("Summary("))

        db_rows = db.get_call_messages(call_id, db_path=self.db_path)
        self.assertEqual(len(db_rows), 12)

    def test_merge_lead_state_preserves_existing_values(self) -> None:
        """Verify lead_state merges new non-empty fields without wiping prior ones."""
        state: dict = {}
        memory.init_conversation_memory(state, db_path=self.db_path)

        memory.update_session_lead_state(state, {"name": "Simran", "need": "Voice AI"})
        memory.update_session_lead_state(state, {"name": None, "budget": "$1000/mo", "timeline": "Q1"})

        self.assertEqual(state["lead_state"]["name"], "Simran")
        self.assertEqual(state["lead_state"]["need"], "Voice AI")
        self.assertEqual(state["lead_state"]["budget"], "$1000/mo")
        self.assertEqual(state["lead_state"]["timeline"], "Q1")


if __name__ == "__main__":
    unittest.main()
