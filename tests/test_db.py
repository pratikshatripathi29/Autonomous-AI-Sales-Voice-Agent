"""Unit tests for the SQLite database layer in core/db.py."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from core import db


class TestDatabaseLayer(unittest.TestCase):
    """Verify table initialization and CRUD helpers across all five tables."""

    def setUp(self) -> None:
        """Create an isolated temporary SQLite database for each test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_sales_agent.db"
        db.init_db(db_path=self.db_path)

    def tearDown(self) -> None:
        """Remove the temporary directory after each test case."""
        self.temp_dir.cleanup()

    def test_init_db_creates_expected_tables(self) -> None:
        """Ensure init_db creates all five schema tables."""
        with db.get_connection(db_path=self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        conn.close()
        table_names = {row["name"] for row in rows}
        expected = {"leads", "calls", "messages", "appointments", "crm_sync_log"}
        self.assertTrue(expected.issubset(table_names))

    def test_lead_crud_and_field_validation(self) -> None:
        """Verify lead creation, retrieval, listing, updating, and invalid field guard."""
        lead_id = db.create_lead(
            name="Asha Verma",
            phone="+91-9876543210",
            email="asha@acme.io",
            company="Acme Corp",
            language="en",
            score=40,
            status="new",
            db_path=self.db_path,
        )
        lead = db.get_lead(lead_id, db_path=self.db_path)
        self.assertIsNotNone(lead)
        assert lead is not None
        self.assertEqual(lead["name"], "Asha Verma")
        self.assertEqual(lead["score"], 40)

        updated = db.update_lead(
            lead_id,
            score=85,
            status="qualified",
            db_path=self.db_path,
        )
        self.assertTrue(updated)
        refreshed = db.get_lead(lead_id, db_path=self.db_path)
        assert refreshed is not None
        self.assertEqual(refreshed["score"], 85)
        self.assertEqual(refreshed["status"], "qualified")

        qualified_leads = db.list_leads(status="qualified", db_path=self.db_path)
        self.assertEqual(len(qualified_leads), 1)

        with self.assertRaises(ValueError):
            db.update_lead(lead_id, db_path=self.db_path, invalid_col="oops")

    def test_call_and_message_lifecycle(self) -> None:
        """Verify creating calls, adding messages in order, and ending calls."""
        lead_id = db.create_lead(name="Ravi Patel", db_path=self.db_path)
        call_id = db.create_call(lead_id=lead_id, db_path=self.db_path)

        msg1_id = db.add_message(
            call_id=call_id,
            role="user",
            text="Tell me about pricing.",
            language="en",
            sentiment=0.2,
            db_path=self.db_path,
        )
        msg2_id = db.add_message(
            call_id=call_id,
            role="assistant",
            text="Our starter plan begins at $49/month.",
            language="en",
            sentiment=0.6,
            db_path=self.db_path,
        )
        self.assertLess(msg1_id, msg2_id)

        messages = db.get_call_messages(call_id, db_path=self.db_path)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

        ended = db.end_call(
            call_id=call_id,
            sentiment_avg=0.4,
            summary="Lead asked about pricing.",
            outcome="follow_up",
            db_path=self.db_path,
        )
        self.assertTrue(ended)

        call_row = db.get_call(call_id, db_path=self.db_path)
        assert call_row is not None
        self.assertIsNotNone(call_row["ended_at"])
        self.assertEqual(call_row["outcome"], "follow_up")
        self.assertAlmostEqual(call_row["sentiment_avg"], 0.4)

    def test_appointments_and_crm_sync_log(self) -> None:
        """Verify appointment slot booking/listing and CRM sync logging."""
        lead_id = db.create_lead(name="Maya Lin", db_path=self.db_path)
        call_id = db.create_call(lead_id=lead_id, db_path=self.db_path)

        appt_id = db.create_appointment(
            lead_id=lead_id,
            call_id=call_id,
            slot_start="2026-11-26T10:00:00+00:00",
            slot_end="2026-11-26T10:30:00+00:00",
            db_path=self.db_path,
        )
        booked = db.get_booked_slots(db_path=self.db_path)
        self.assertEqual(len(booked), 1)
        self.assertEqual(booked[0]["id"], appt_id)

        self.assertTrue(
            db.update_appointment_status(appt_id, "cancelled", db_path=self.db_path)
        )
        self.assertEqual(len(db.get_booked_slots(db_path=self.db_path)), 0)

        log_id = db.log_crm_sync(
            lead_id=lead_id,
            payload='{"lead_id": 1, "status": "qualified"}',
            status="success",
            db_path=self.db_path,
        )
        logs = db.get_crm_sync_logs(lead_id=lead_id, db_path=self.db_path)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["id"], log_id)

    def test_foreign_key_enforcement(self) -> None:
        """Verify SQLite rejects orphan messages referencing a nonexistent call."""
        with self.assertRaises(sqlite3.IntegrityError):
            db.add_message(
                call_id=99999,
                role="user",
                text="Orphan message",
                db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()

