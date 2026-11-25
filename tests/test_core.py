"""Pytest suite for Stage 8: lead scoring, slot booking conflicts, and JSON parsing."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from core import booking, db, leads, llm, summary


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Provide an isolated SQLite database per test."""
    db_file = tmp_path / "pytest_core.db"
    db.init_db(db_path=db_file)
    return db_file


# --- Lead Scoring Tests ---


@pytest.mark.parametrize(
    ("lead_state", "sentiments", "intents", "expected_score", "expected_status"),
    [
        (
            {
                "budget": "$500/mo",
                "timeline": "3 weeks",
                "company": "Acme Corp",
                "email": "cto@acme.io",
                "wants_booking": True,
            },
            [0.6, 0.8],
            ["pricing"],
            100,
            "Hot",
        ),
        (
            {
                "budget": "$200/mo",
                "timeline": "2 months",
                "email": "lead@example.com",
            },
            [-0.4],
            ["inquiry"],
            40,
            "Warm",
        ),
        (
            {
                "timeline": "6 months",
                "company": "Slow Co",
            },
            [0.0],
            ["inquiry"],
            15,
            "Cold",
        ),
        (
            {},
            [-0.9, -0.5],
            ["complaint"],
            0,
            "Cold",
        ),
    ],
)
def test_score_lead_rules_and_clamping(
    lead_state: dict,
    sentiments: list[float],
    intents: list[str],
    expected_score: int,
    expected_status: str,
) -> None:
    """Verify rule-based weights, Hot/Warm/Cold boundaries, and 0-100 clamping."""
    score, status = leads.score_lead(lead_state, sentiments, intents)
    assert score == expected_score
    assert status == expected_status
    breakdown = leads.get_score_breakdown(lead_state, sentiments, intents)
    assert isinstance(breakdown, list)


# --- Slot Booking & Conflict Tests ---


def test_booking_conflict_returns_three_alternatives(temp_db: Path) -> None:
    """Double-booking a slot fails atomically and returns 3 nearest open slots."""
    lead_id = db.create_lead(name="Asha", email="asha@test.com", db_path=temp_db)
    call_id = db.create_call(lead_id=lead_id, db_path=temp_db)
    now_ist = datetime(2026, 11, 25, 10, 0, tzinfo=booking.IST_ZONE)
    target_slot = datetime(2026, 11, 26, 14, 0, tzinfo=booking.IST_ZONE)

    first = booking.book_slot(
        lead_id=lead_id,
        call_id=call_id,
        slot_start=target_slot,
        attendee_name="Asha",
        attendee_email="asha@test.com",
        now=now_ist,
        db_path=temp_db,
    )
    assert first["booked"] is True
    assert "BEGIN:VCALENDAR" in first["ics"]

    conflict = booking.book_slot(
        lead_id=lead_id,
        call_id=call_id,
        slot_start=target_slot,
        now=now_ist,
        db_path=temp_db,
    )
    assert conflict["booked"] is False
    assert len(conflict["alternatives"]) == 3
    assert all(alt["slot_start"] != first["slot_start"] for alt in conflict["alternatives"])


def test_booking_rejects_past_outside_hours_and_bad_format(temp_db: Path) -> None:
    """Verify past times, weekend/after-hours slots, and malformed strings never crash."""
    lead_id = db.create_lead(name="Raj", phone="+91-9999999999", db_path=temp_db)
    now_ist = datetime(2026, 11, 25, 14, 0, tzinfo=booking.IST_ZONE)

    past_res = booking.book_slot(
        lead_id=lead_id,
        call_id=None,
        slot_start=datetime(2026, 11, 24, 11, 0, tzinfo=booking.IST_ZONE),
        now=now_ist,
        db_path=temp_db,
    )
    assert past_res["booked"] is False
    assert len(past_res["alternatives"]) == 3

    # Saturday 11:00 AM is outside Mon-Fri business hours.
    weekend_res = booking.book_slot(
        lead_id=lead_id,
        call_id=None,
        slot_start=datetime(2026, 11, 28, 11, 0, tzinfo=booking.IST_ZONE),
        now=now_ist,
        db_path=temp_db,
    )
    assert weekend_res["booked"] is False
    assert "outside business hours" in weekend_res["reason"]

    invalid_res = booking.book_slot(
        lead_id=lead_id,
        call_id=None,
        slot_start="not-an-iso-date",
        now=now_ist,
        db_path=temp_db,
    )
    assert invalid_res["booked"] is False
    assert len(invalid_res["alternatives"]) == 3


def test_available_slots_and_ist_parsing(temp_db: Path) -> None:
    """Verify slot generation excludes weekends and dateparser parses IST times."""
    now_ist = datetime(2026, 11, 25, 9, 0, tzinfo=booking.IST_ZONE)
    parsed = booking.parse_preferred_time("tomorrow at 4pm", now=now_ist)
    assert parsed is not None
    assert parsed.date() == date(2026, 11, 26)
    assert (parsed.hour, parsed.minute) == (16, 0)

    weekend_slots = booking.get_available_slots(
        date_range=(date(2026, 11, 28), date(2026, 11, 29)),
        now=now_ist,
        db_path=temp_db,
    )
    assert weekend_slots == []


# --- JSON Parsing Tests ---


def test_parse_turn_json_handles_fences_prose_and_malformed() -> None:
    """Verify JSON parser strips code fences, extracts embedded JSON, and falls back safely."""
    fenced = """```json
    {
      "reply": "Namaste! How can I help?",
      "language": "HI",
      "sentiment": "positive",
      "sentiment_score": 4.2,
      "intent": "BOOKING",
      "extracted": {"name": "  Kabir  ", "budget": ""},
      "wants_booking": true,
      "preferred_time": "Friday 11 AM"
    }
    ```"""
    parsed = llm._parse_turn_json(fenced, default_lang="en")
    assert parsed["reply"] == "Namaste! How can I help?"
    assert parsed["language"] == "hi"
    assert parsed["sentiment_score"] == 1.0
    assert parsed["intent"] == "booking"
    assert parsed["extracted"]["name"] == "Kabir"
    assert parsed["extracted"]["budget"] is None
    assert parsed["wants_booking"] is True

    prose_wrapped = 'Sure, here is the response: {"reply": "Demo booked!", "sentiment": "invalid", "intent": "unknown"} Thanks!'
    embedded = llm._parse_turn_json(prose_wrapped, default_lang="en")
    assert embedded["reply"] == "Demo booked!"
    assert embedded["sentiment"] == "neutral"
    assert embedded["intent"] == "other"

    plain_fallback = llm._parse_turn_json("Sorry, I will connect a human specialist.", default_lang="pa")
    assert plain_fallback["reply"] == "Sorry, I will connect a human specialist."
    assert plain_fallback["language"] == "pa"
    assert plain_fallback["wants_booking"] is False


def test_parse_call_summary_handles_json_and_legacy_text() -> None:
    """Verify summary parser handles structured JSON, legacy plain strings, and empty input."""
    json_str = '{"summary": "Booked demo", "objections": ["Price"], "outcome": "booked"}'
    parsed_json = summary.parse_call_summary(json_str)
    assert parsed_json["summary"] == "Booked demo"
    assert parsed_json["objections"] == ["Price"]
    assert parsed_json["outcome"] == "booked"

    legacy = summary.parse_call_summary("Plain legacy summary text.")
    assert legacy["summary"] == "Plain legacy summary text."
    assert legacy["objections"] == []

    empty = summary.parse_call_summary("")
    assert empty["summary"] == ""

