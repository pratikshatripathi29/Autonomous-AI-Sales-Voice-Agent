"""Unit tests for Stage 7 post-call summarization, analytics KPIs, charts, and AI insights."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from core import analytics, db, summary


class TestSummaryAndAnalytics(unittest.TestCase):
    """Verify Gemini call summary persistence, KPI calculation, chart builders, and insights."""

    def setUp(self) -> None:
        """Create an isolated temporary SQLite database with sample call/lead/appointment data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_summary_analytics.db"
        db.init_db(db_path=self.db_path)

        self.lead_hot = db.create_lead(
            name="Arjun Mehta",
            email="arjun@mehta.io",
            language="en",
            score=85,
            status="Hot",
            db_path=self.db_path,
        )
        self.lead_warm = db.create_lead(
            name="Priya Kaur",
            phone="+91-9876500001",
            language="hi",
            score=50,
            status="Warm",
            db_path=self.db_path,
        )

        self.call_1 = db.create_call(lead_id=self.lead_hot, db_path=self.db_path)
        db.add_message(self.call_1, "user", "We miss inbound leads after hours.", "en", 0.3, db_path=self.db_path)
        db.add_message(self.call_1, "assistant", "Our AI receptionist answers 24/7.", "en", None, db_path=self.db_path)
        db.add_message(self.call_1, "user", "Price is a bit high, can we see a demo?", "en", 0.5, db_path=self.db_path)

        self.call_2 = db.create_call(lead_id=self.lead_warm, db_path=self.db_path)
        db.add_message(self.call_2, "user", "Hume Hindi support chahiye.", "hi", 0.2, db_path=self.db_path)

        db.create_appointment(
            lead_id=self.lead_hot,
            call_id=self.call_1,
            slot_start="2026-11-26T10:00:00+05:30",
            slot_end="2026-11-26T10:30:00+05:30",
            db_path=self.db_path,
        )

    def tearDown(self) -> None:
        """Clean up temporary database files."""
        self.temp_dir.cleanup()

    def test_summarize_call_saves_structured_json_to_calls_table(self) -> None:
        """Verify summarize_call calls Gemini, normalizes JSON, and updates calls row."""
        mock_client = MagicMock()
        mock_client.invoke.return_value = SimpleNamespace(
            content=json.dumps(
                {
                    "summary": "Arjun wants 24/7 voice coverage and requested a demo.",
                    "key_points": ["Needs after-hours coverage", "Interested in Growth plan"],
                    "customer_pain_points": ["Missing after-hours calls"],
                    "objections": ["Price concern"],
                    "next_steps": ["Run Thursday demo"],
                    "outcome": "booked",
                    "overall_sentiment": "positive",
                    "recommended_action": "Send calendar invite and pricing sheet.",
                }
            )
        )

        with patch.object(summary, "GEMINI_API_KEY", "test-key"), patch.object(
            summary, "_create_chat_client", return_value=mock_client
        ):
            result = summary.summarize_call(self.call_1, db_path=self.db_path)

        self.assertEqual(result["outcome"], "booked")
        self.assertEqual(result["objections"], ["Price concern"])

        saved_call = db.get_call(self.call_1, db_path=self.db_path)
        assert saved_call is not None
        self.assertIsNotNone(saved_call["ended_at"])
        self.assertEqual(saved_call["outcome"], "booked")
        self.assertAlmostEqual(saved_call["sentiment_avg"], 0.4)
        parsed_saved = summary.parse_call_summary(saved_call["summary"])
        self.assertEqual(parsed_saved["customer_pain_points"], ["Missing after-hours calls"])

    def test_analytics_kpis_charts_and_insights(self) -> None:
        """Verify KPI math, summary aggregation, all 6 Plotly figures, and AI insights."""
        db.end_call(
            self.call_1,
            sentiment_avg=0.4,
            summary=json.dumps(
                {
                    "summary": "Demo booked.",
                    "customer_pain_points": ["Slow lead response"],
                    "objections": ["Setup effort"],
                    "outcome": "booked",
                }
            ),
            outcome="booked",
            db_path=self.db_path,
        )

        data = analytics.load_analytics_data(db_path=self.db_path)
        kpis = analytics.compute_kpis(data)
        self.assertEqual(kpis["total_calls"], 2)
        self.assertEqual(kpis["total_leads"], 2)
        self.assertEqual(kpis["hot_leads_pct"], 50.0)
        self.assertEqual(kpis["appointments_booked"], 1)
        self.assertEqual(kpis["conversion_rate"], 50.0)

        issues = analytics.aggregate_objections_and_pain_points(data["calls"])
        self.assertEqual(issues["objections"], [("Setup effort", 1)])
        self.assertEqual(issues["pain_points"], [("Slow lead response", 1)])

        figs = [
            analytics.build_conversion_funnel(kpis),
            analytics.build_lead_status_pie(data["leads"]),
            analytics.build_sentiment_over_time_chart(data["calls"], data["messages"]),
            analytics.build_calls_per_day_chart(data["calls"]),
            analytics.build_language_distribution_chart(data["messages"], data["leads"]),
            analytics.build_top_issues_chart(issues),
        ]
        for fig in figs:
            self.assertGreaterEqual(len(fig.data), 1)

        mock_client = MagicMock()
        mock_client.invoke.return_value = SimpleNamespace(
            content='{"insights": ["50% of leads are Hot; prioritize fast demo follow-up.", "Address setup effort early in pitch.", "Expand Hindi onboarding collateral."]}'
        )
        with patch.object(analytics, "GEMINI_API_KEY", "test-key"), patch.object(
            analytics, "_create_chat_client", return_value=mock_client
        ):
            insights = analytics.generate_business_insights(data)

        self.assertEqual(len(insights), 3)
        self.assertIn("50%", insights[0])


if __name__ == "__main__":
    unittest.main()

