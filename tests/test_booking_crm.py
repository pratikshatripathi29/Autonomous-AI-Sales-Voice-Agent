"""Unit tests for Stage 6 booking engine and pluggable CRM sync."""

from __future__ import annotations

import csv
from datetime import date, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core import booking, crm, db


class TestBookingAndCRM(unittest.TestCase):
    """Verify slot generation, dateparser IST parsing, double-booking guard, ICS, and CRM sync."""

    def setUp(self) -> None:
        """Create an isolated temporary SQLite database and CSV export path."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test_booking_crm.db"
        self.csv_path = self.root / "crm_export.csv"
        db.init_db(db_path=self.db_path)

        # Wednesday Nov 25, 2026 at 11:00 AM IST
        self.now_ist = datetime(2026, 11, 25, 11, 0, tzinfo=booking.IST_ZONE)
        self.lead_id = db.create_lead(
            name="Neha Kapoor",
            phone="+91-9811122233",
            email="neha@example.com",
            company="Kapoor Retail",
            score=80,
            status="Hot",
            db_path=self.db_path,
        )
        self.call_id = db.create_call(lead_id=self.lead_id, db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_parse_preferred_time_relative_to_ist(self) -> None:
        """Verify dateparser resolves relative phrases in Asia/Kolkata."""
        parsed = booking.parse_preferred_time("tomorrow at 3pm", now=self.now_ist)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.date(), date(2026, 11, 26))
        self.assertEqual((parsed.hour, parsed.minute), (15, 0))
        self.assertEqual(parsed.utcoffset().total_seconds(), 5.5 * 3600)

    def test_available_slots_exclude_weekends_and_booked_slots(self) -> None:
        """Verify Mon-Fri 10:00-17:00 slots exclude weekends and existing bookings."""
        # Nov 27 (Fri) through Nov 29 (Sun)
        slots_before = booking.get_available_slots(
            date_range=(date(2026, 11, 27), date(2026, 11, 29)),
            now=self.now_ist,
            db_path=self.db_path,
        )
        # Friday has 14 slots (10:00 .. 16:30); Sat & Sun have 0.
        self.assertEqual(len(slots_before), 14)

        target_slot = slots_before[0]["slot_start"]
        booked = booking.book_slot(
            lead_id=self.lead_id,
            call_id=self.call_id,
            slot_start=target_slot,
            attendee_name="Neha Kapoor",
            attendee_email="neha@example.com",
            now=self.now_ist,
            db_path=self.db_path,
        )
        self.assertTrue(booked["booked"])
        self.assertIn("BEGIN:VCALENDAR", booked["ics"])
        self.assertIn("DTSTART:", booked["ics"])

        slots_after = booking.get_available_slots(
            date_range=(date(2026, 11, 27), date(2026, 11, 29)),
            now=self.now_ist,
            db_path=self.db_path,
        )
        self.assertEqual(len(slots_after), 13)

    def test_double_booking_returns_three_nearest_alternatives(self) -> None:
        """Verify attempting to book an already-booked slot returns 3 alternatives."""
        slot_dt = datetime(2026, 11, 26, 14, 0, tzinfo=booking.IST_ZONE)
        first = booking.book_slot(
            lead_id=self.lead_id,
            call_id=self.call_id,
            slot_start=slot_dt,
            now=self.now_ist,
            db_path=self.db_path,
        )
        self.assertTrue(first["booked"])

        second = booking.book_slot(
            lead_id=self.lead_id,
            call_id=self.call_id,
            slot_start=slot_dt,
            now=self.now_ist,
            db_path=self.db_path,
        )
        self.assertFalse(second["booked"])
        self.assertEqual(len(second["alternatives"]), 3)
        self.assertNotIn(first["slot_start"], [a["slot_start"] for a in second["alternatives"]])

    def test_process_booking_request_requires_contact_details(self) -> None:
        """Verify booking asks for name/contact first if missing."""
        missing_contact = booking.process_booking_request(
            lead_id=None,
            call_id=self.call_id,
            lead_state={"name": "Neha", "phone": None, "email": None},
            preferred_time_text="tomorrow at 3pm",
            now=self.now_ist,
            db_path=self.db_path,
        )
        self.assertEqual(missing_contact["status"], "need_contact")
        self.assertIn("phone number or email", missing_contact["message"])

    def test_crm_sync_local_and_webhook_modes(self) -> None:
        """Verify local CRM mode writes SQLite + CSV and webhook mode POSTs with retries."""
        slot_dt = datetime(2026, 11, 26, 15, 0, tzinfo=booking.IST_ZONE)
        booking.book_slot(
            lead_id=self.lead_id,
            call_id=self.call_id,
            slot_start=slot_dt,
            now=self.now_ist,
            db_path=self.db_path,
        )
        db.end_call(
            self.call_id,
            sentiment_avg=0.65,
            summary="Lead wants Growth plan demo on Thursday.",
            outcome="booked",
            db_path=self.db_path,
        )

        local_res = crm.sync_lead(
            {"id": self.lead_id},
            mode="local",
            db_path=self.db_path,
            csv_path=self.csv_path,
        )
        self.assertTrue(local_res["ok"])
        self.assertEqual(local_res["status"], "synced_local")
        self.assertTrue(self.csv_path.exists())

        with self.csv_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Neha Kapoor")
        self.assertEqual(rows[0]["score"], "80")

        logs = db.get_crm_sync_logs(lead_id=self.lead_id, db_path=self.db_path)
        self.assertEqual(len(logs), 1)
        parsed_payload = json.loads(logs[0]["payload"])
        self.assertEqual(parsed_payload["summary"], "Lead wants Growth plan demo on Thursday.")
        self.assertIsNotNone(parsed_payload["appointment"])

        # Webhook mode retries once on 500 then succeeds on 200.
        fail_resp = MagicMock(status_code=500)
        ok_resp = MagicMock(status_code=200)
        with patch.object(crm.requests, "post", side_effect=[fail_resp, ok_resp]) as mock_post:
            webhook_res = crm.sync_lead(
                {"id": self.lead_id},
                mode="webhook",
                db_path=self.db_path,
                webhook_url="https://crm.example.com/hook",
                max_retries=3,
            )
        self.assertTrue(webhook_res["ok"])
        self.assertEqual(webhook_res["status"], "synced_webhook")
        self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()

