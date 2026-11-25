"""Unit tests for Stage 4 RAG indexing, retrieval, and query rewriting."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from core import llm, rag


class _DeterministicFakeEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for offline FAISS unit tests."""

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("price") + lowered.count("starter") + lowered.count("$49")),
            float(lowered.count("trial") + lowered.count("cancel") + lowered.count("14-day")),
            float(lowered.count("security") + lowered.count("soc 2") + lowered.count("tls")),
            float(len(lowered) % 17 + 1),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class TestRAGPipeline(unittest.TestCase):
    """Verify FAISS build/load/retrieve, formatting, and contextual query rewriting."""

    def setUp(self) -> None:
        """Create temporary knowledge and FAISS index directories with sample docs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.knowledge_dir = self.root / "knowledge"
        self.index_dir = self.root / "faiss_index"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        (self.knowledge_dir / "pricing.md").write_text(
            "Starter Plan costs $49/month for 500 voice minutes and 1 phone number.",
            encoding="utf-8",
        )
        (self.knowledge_dir / "trial.txt").write_text(
            "Every customer gets a 14-day free trial with zero cancellation fees.",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        """Remove temporary test artifacts."""
        self.temp_dir.cleanup()

    def test_build_load_and_retrieve_faiss_index(self) -> None:
        """Verify build_index saves FAISS locally and load_index + retrieve return ranked hits."""
        fake_embeddings = _DeterministicFakeEmbeddings()
        with patch.object(rag, "GEMINI_API_KEY", "test-key"), patch.object(
            rag, "_create_embeddings", return_value=fake_embeddings
        ):
            built = rag.build_index(
                knowledge_dir=self.knowledge_dir,
                index_dir=self.index_dir,
            )
            self.assertIsNotNone(built)
            self.assertTrue((self.index_dir / "index.faiss").exists())
            self.assertTrue((self.index_dir / "index.pkl").exists())

            hits = rag.retrieve(
                "Starter price $49",
                k=2,
                knowledge_dir=self.knowledge_dir,
                index_dir=self.index_dir,
            )

        self.assertEqual(len(hits), 2)
        top_text, top_source, top_score = hits[0]
        self.assertIn("$49", top_text)
        self.assertEqual(top_source, "pricing.md")
        self.assertIsInstance(top_score, float)

        formatted = rag.format_context(hits)
        self.assertIn("[Source: pricing.md", formatted)
        self.assertIn("Starter Plan", formatted)

    def test_load_index_builds_when_missing(self) -> None:
        """Verify load_index delegates to build_index when no saved index exists."""
        fake_embeddings = _DeterministicFakeEmbeddings()
        with patch.object(rag, "GEMINI_API_KEY", "test-key"), patch.object(
            rag, "_create_embeddings", return_value=fake_embeddings
        ):
            loaded = rag.load_index(
                knowledge_dir=self.knowledge_dir,
                index_dir=self.index_dir,
            )
        self.assertIsNotNone(loaded)
        self.assertTrue((self.index_dir / "index.faiss").exists())

    def test_rewrite_search_query_uses_last_user_turn_and_intent(self) -> None:
        """Verify short follow-ups like 'how much?' rewrite using prior turn + intent."""
        history = [
            {"role": "user", "text": "Tell me about AI Voice Receptionist"},
            {"role": "assistant", "text": "It answers inbound calls 24/7."},
        ]
        mock_client = MagicMock()
        mock_client.invoke.return_value = SimpleNamespace(
            content="AI Voice Receptionist pricing plans cost"
        )

        with patch.object(llm, "GEMINI_API_KEY", "test-key"), patch.object(
            llm, "_create_chat_client", return_value=mock_client
        ):
            rewritten = llm.rewrite_search_query(
                "how much?",
                history=history,
                last_intent="inquiry",
            )

        self.assertEqual(rewritten, "AI Voice Receptionist pricing plans cost")
        prompt_sent = mock_client.invoke.call_args.args[0][0].content
        self.assertIn("AI Voice Receptionist", prompt_sent)
        self.assertIn("inquiry", prompt_sent)

        # Fallback path when Gemini fails should still include prior user turn + intent.
        mock_client.invoke.side_effect = RuntimeError("Network error")
        with patch.object(llm, "GEMINI_API_KEY", "test-key"), patch.object(
            llm, "_create_chat_client", return_value=mock_client
        ):
            fallback = llm.rewrite_search_query(
                "how much?",
                history=history,
                last_intent="pricing",
            )
        self.assertIn("AI Voice Receptionist", fallback)
        self.assertIn("pricing", fallback)
        self.assertIn("how much?", fallback)


if __name__ == "__main__":
    unittest.main()

