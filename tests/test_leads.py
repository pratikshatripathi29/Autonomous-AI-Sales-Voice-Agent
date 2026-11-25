"""Unit tests for Stage 5 lead scoring, breakdown, and SQLite lead sync."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from core import analytics, db, leads, memory


class TestLeadScoringAndSync(unittest.TestCase):
    """Verify rule-based lead scoring, clamping, breakdowns, SQLite sync, and charts."""

    def setUp(self) -> None:
        """Create an isolated temporary SQLite database."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_leads.db"
        db.init_db(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database directory."""
        self.temp_dir.cleanup()

    def test_hot_lead_scoring_and_breakdown(self) -> None:
        """Verify Hot lead reaches >=70 and returns transparent rule reasons."""
        lead_state = {
            "name": "Asha Verma",
            "email": "asha@acme.io",
            "company": "Acme Corp (50 reps)",
            "budget": "$500/month",
            "timeline": "next month",
            "wants_booking": True,
        }
        score, status = leads.score_lead(
            lead_state,
            sentiment_history=[0.5, 0.7],
            intents=["inquiry", "pricing"],
        )
        self.assertEqual(score, 100)
        self.assertEqual(status, "Hot")

        breakdown = leads.get_score_breakdown(
            lead_state,
            sentiment_history=[0.5, 0.7],
            intents=["inquiry", "pricing"],
        )
        self.assertEqual(len(breakdown), 6)
        self.assertTrue(any("Budget provided" in item for item in breakdown))
        self.assertTrue(any("Timeline under 3 months" in item for item in breakdown))

    def test_warm_cold_and_negative_sentiment_penalty_with_clamp(self) -> None:
        """Verify Warm/Cold thresholds, negative sentiment -15 penalty, and 0-100 clamp."""
        warm_state = {
            "budget": "$200/mo",
            "timeline": "6 weeks",
            "email": "bob@test.com",
        }
        # 20 (budget) + 20 (<3mo) + 15 (contact) - 15 (negative sentiment) = 40 -> Warm
        score, status = leads.score_lead(
            warm_state,
            sentiment_history=[-0.4, -0.2],
            intents=["inquiry"],
        )
        self.assertEqual(score, 40)
        self.assertEqual(status, "Warm")

        # Empty lead with negative sentiment clamps to 0 (not -15) -> Cold
        cold_score, cold_status = leads.score_lead(
            {},
            sentiment_history=[-0.8],
            intents=["complaint"],
        )
        self.assertEqual(cold_score, 0)
        self.assertEqual(cold_status, "Cold")

    def test_sync_lead_to_db_creates_on_first_contact_and_updates_afterward(self) -> None:
        """Verify lead row is created once name/phone/email appears and updated on later turns."""
        state: dict = {}
        call_id = memory.init_conversation_memory(state, db_path=self.db_path)

        # Turn 1: only budget mentioned, no name/phone/email yet -> should not create lead row.
        memory.update_session_lead_state(state, {"budget": "$300/mo"})
        created_early = leads.sync_lead_to_db(state, score=20, status="Cold", db_path=self.db_path)
        self.assertIsNone(created_early)
        self.assertEqual(len(db.list_leads(db_path=self.db_path)), 0)

        # Turn 2: name & phone captured -> creates lead and links active call.
        memory.update_session_lead_state(
            state,
            {"name": "Kabir Singh", "phone": "+91-9988776655"},
        )
        lead_id = leads.sync_lead_to_db(
            state,
            score=45,
            status="Warm",
            language="pa",
            db_path=self.db_path,
        )
        self.assertIsNotNone(lead_id)
        assert lead_id is not None
        call_row = db.get_call(call_id, db_path=self.db_path)
        assert call_row is not None
        self.assertEqual(call_row["lead_id"], lead_id)

        # Turn 3: score increases -> updates same lead row without creating duplicates.
        memory.update_session_lead_state(state, {"company": "Singh Logistics"}, wants_booking=True)
        updated_id = leads.sync_lead_to_db(
            state,
            score=85,
            status="Hot",
            language="pa",
            db_path=self.db_path,
        )
        self.assertEqual(updated_id, lead_id)
        all_leads = db.list_leads(db_path=self.db_path)
        self.assertEqual(len(all_leads), 1)
        self.assertEqual(all_leads[0]["score"], 85)
        self.assertEqual(all_leads[0]["status"], "Hot")

    def test_plotly_live_figures_build_cleanly(self) -> None:
        """Verify Plotly gauge and sentiment trend charts construct valid figures."""
        gauge = analytics.build_score_gauge(75, "Hot")
        trend = analytics.build_sentiment_trend_figure([0.1, -0.2, 0.6])
        self.assertEqual(len(gauge.data), 1)
        self.assertEqual(len(trend.data), 1)


if __name__ == "__main__":
    unittest.main()

