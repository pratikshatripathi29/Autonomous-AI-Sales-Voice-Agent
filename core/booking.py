"""Appointment scheduling, natural-language time parsing, double-booking guard, and ICS export."""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import dateparser

from core.config import COMPANY_NAME, TIMEZONE_NAME
from core.db import get_booked_slots, get_connection

logger = logging.getLogger(__name__)

IST_ZONE = ZoneInfo(TIMEZONE_NAME)
BUSINESS_START_HOUR = 10
BUSINESS_END_HOUR = 17
SLOT_DURATION_MINUTES = 30


def _now_ist() -> datetime:
    """Return the current datetime in the configured Asia/Kolkata timezone."""
    return datetime.now(IST_ZONE)


def _to_ist(dt: datetime) -> datetime:
    """Normalize a naive or aware datetime into the Asia/Kolkata timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST_ZONE)
    return dt.astimezone(IST_ZONE)


def _parse_iso_dt(value: datetime | str) -> datetime:
    """Convert an ISO string or datetime object into an IST-aware datetime."""
    if isinstance(value, datetime):
        return _to_ist(value)
    return _to_ist(datetime.fromisoformat(value))


def _format_slot_label(slot_dt: datetime) -> str:
    """Format an IST datetime into a voice- and UI-friendly label."""
    return slot_dt.strftime("%a, %b %d at %I:%M %p IST").replace(" 0", " ")


def _is_business_slot(slot_dt: datetime) -> bool:
    """Return True if slot_dt falls on Mon-Fri between 10:00 and 16:30."""
    if slot_dt.weekday() >= 5:
        return False
    slot_time = slot_dt.time()
    start_bound = time(BUSINESS_START_HOUR, 0)
    last_start = time(BUSINESS_END_HOUR - 1, 30)
    return start_bound <= slot_time <= last_start and slot_dt.minute in (0, 30)


def parse_preferred_time(
    text: str,
    now: datetime | None = None,
) -> datetime | None:
    """Parse natural-language date/time text relative to Asia/Kolkata and snap to 30-min slot."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    base_now = _to_ist(now) if now is not None else _now_ist()
    # dateparser expects a naive RELATIVE_BASE matching the configured TIMEZONE setting.
    settings = {
        "TIMEZONE": TIMEZONE_NAME,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base_now.replace(tzinfo=None),
    }
    try:
        parsed = dateparser.parse(cleaned, settings=settings)
    except Exception:
        logger.exception("dateparser failed on preferred_time='%s'.", cleaned)
        return None

    if parsed is None:
        return None

    parsed_ist = _to_ist(parsed)
    snapped_minute = 0 if parsed_ist.minute < 30 else 30
    return parsed_ist.replace(minute=snapped_minute, second=0, microsecond=0)


def get_available_slots(
    date_range: tuple[date, date] | int = 7,
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Generate available Mon-Fri 10:00-17:00 30-min slots minus SQLite booked slots."""
    base_now = _to_ist(now) if now is not None else _now_ist()

    if isinstance(date_range, int):
        start_date = base_now.date()
        end_date = start_date + timedelta(days=max(0, date_range - 1))
    else:
        start_date, end_date = date_range

    booked_rows = get_booked_slots(status="booked", db_path=db_path)
    booked_starts = {
        _parse_iso_dt(row["slot_start"]).isoformat()
        for row in booked_rows
        if row.get("slot_start")
    }

    slots: list[dict[str, Any]] = []
    current_date = start_date
    while current_date <= end_date:
        # Skip Saturday (5) and Sunday (6).
        if current_date.weekday() < 5:
            slot_dt = datetime.combine(
                current_date,
                time(BUSINESS_START_HOUR, 0),
                tzinfo=IST_ZONE,
            )
            day_end = datetime.combine(
                current_date,
                time(BUSINESS_END_HOUR, 0),
                tzinfo=IST_ZONE,
            )
            while slot_dt < day_end:
                slot_end = slot_dt + timedelta(minutes=SLOT_DURATION_MINUTES)
                iso_start = slot_dt.isoformat()
                if slot_dt > base_now and iso_start not in booked_starts:
                    slots.append(
                        {
                            "slot_start": iso_start,
                            "slot_end": slot_end.isoformat(),
                            "label": _format_slot_label(slot_dt),
                        }
                    )
                slot_dt = slot_end
        current_date += timedelta(days=1)

    return slots


def generate_ics(
    slot_start: datetime | str,
    slot_end: datetime | str,
    summary: str = f"{COMPANY_NAME} Product Demo",
    attendee_name: str = "",
    attendee_email: str = "",
    uid: str | None = None,
) -> str:
    """Build an RFC 5545 .ics calendar invite string for the booked appointment."""
    start_utc = _parse_iso_dt(slot_start).astimezone(timezone.utc)
    end_utc = _parse_iso_dt(slot_end).astimezone(timezone.utc)
    stamp_utc = datetime.now(timezone.utc)
    event_uid = uid or f"{start_utc.strftime('%Y%m%dT%H%M%SZ')}-sales-agent@novacloud.ai"

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:-//{COMPANY_NAME}//Sales Voice Agent//EN",
        "BEGIN:VEVENT",
        f"UID:{event_uid}",
        f"DTSTAMP:{stamp_utc.strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{start_utc.strftime('%Y%m%dT%H%M%SZ')}",
        f"DTEND:{end_utc.strftime('%Y%m%dT%H%M%SZ')}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:Demo scheduled with {attendee_name or 'Prospective Customer'} ({attendee_email or 'no email'})",
        "STATUS:CONFIRMED",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def _nearest_alternatives(
    requested_dt: datetime,
    now: datetime | None = None,
    count: int = 3,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return the nearest available business-hour slots sorted by closeness to requested_dt."""
    base_now = _to_ist(now) if now is not None else _now_ist()
    start_day = min(base_now.date(), requested_dt.date())
    end_day = max(base_now.date(), requested_dt.date()) + timedelta(days=10)
    candidates = get_available_slots(
        date_range=(start_day, end_day),
        now=base_now,
        db_path=db_path,
    )
    ranked = sorted(
        candidates,
        key=lambda item: abs(
            (_parse_iso_dt(item["slot_start"]) - requested_dt).total_seconds()
        ),
    )
    return ranked[:count]


def book_slot(
    lead_id: int,
    call_id: int | None,
    slot_start: datetime | str,
    attendee_name: str = "",
    attendee_email: str = "",
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Book a 30-min slot atomically; if unavailable, return the 3 nearest alternatives."""
    base_now = _to_ist(now) if now is not None else _now_ist()
    try:
        requested_dt = _parse_iso_dt(slot_start)
    except ValueError:
        logger.warning("Invalid slot_start format passed to book_slot: %r", slot_start)
        return {
            "booked": False,
            "reason": "Invalid date/time format.",
            "requested_slot": str(slot_start),
            "alternatives": _nearest_alternatives(base_now, now=base_now, db_path=db_path),
        }

    snapped_minute = 0 if requested_dt.minute < 30 else 30
    requested_dt = requested_dt.replace(minute=snapped_minute, second=0, microsecond=0)
    requested_end = requested_dt + timedelta(minutes=SLOT_DURATION_MINUTES)

    if requested_dt <= base_now:
        alternatives = _nearest_alternatives(base_now, now=base_now, db_path=db_path)
        return {
            "booked": False,
            "reason": "Requested time is in the past.",
            "requested_slot": requested_dt.isoformat(),
            "alternatives": alternatives,
        }

    if not _is_business_slot(requested_dt):
        alternatives = _nearest_alternatives(requested_dt, now=base_now, db_path=db_path)
        return {
            "booked": False,
            "reason": "Requested time is outside business hours (Mon-Fri, 10:00 AM - 5:00 PM IST).",
            "requested_slot": requested_dt.isoformat(),
            "alternatives": alternatives,
        }

    start_iso = requested_dt.isoformat()
    end_iso = requested_end.isoformat()
    created_at = datetime.now(timezone.utc).isoformat()

    with closing(get_connection(db_path)) as conn:
        # BEGIN IMMEDIATE acquires a write lock before checking conflicts so two
        # concurrent callers cannot double-book the exact same slot.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conflict = conn.execute(
                """
                SELECT id FROM appointments
                WHERE status = 'booked'
                  AND slot_start < ?
                  AND slot_end > ?
                LIMIT 1
                """,
                (end_iso, start_iso),
            ).fetchone()

            if conflict is not None:
                conn.rollback()
                alternatives = _nearest_alternatives(requested_dt, now=now, db_path=db_path)
                return {
                    "booked": False,
                    "reason": "That time slot is already booked.",
                    "requested_slot": start_iso,
                    "alternatives": alternatives,
                }

            cursor = conn.execute(
                """
                INSERT INTO appointments (lead_id, call_id, slot_start, slot_end, status, created_at)
                VALUES (?, ?, ?, ?, 'booked', ?)
                """,
                (lead_id, call_id, start_iso, end_iso, created_at),
            )
            conn.commit()
            appt_id = int(cursor.lastrowid)
        except Exception:
            conn.rollback()
            raise

    ics_text = generate_ics(
        slot_start=start_iso,
        slot_end=end_iso,
        attendee_name=attendee_name,
        attendee_email=attendee_email,
        uid=f"appt-{appt_id}@novacloud.ai",
    )
    return {
        "booked": True,
        "appointment_id": appt_id,
        "slot_start": start_iso,
        "slot_end": end_iso,
        "label": _format_slot_label(requested_dt),
        "ics": ics_text,
    }


def process_booking_request(
    lead_id: int | None,
    call_id: int | None,
    lead_state: dict[str, Any] | None,
    preferred_time_text: str | None,
    user_text: str = "",
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Coordinate contact validation, time parsing, booking, and voice-friendly responses."""
    state = lead_state or {}
    name = str(state.get("name") or "").strip()
    phone = str(state.get("phone") or "").strip()
    email = str(state.get("email") or "").strip()

    if not name or not (phone or email) or lead_id is None:
        if not name and not (phone or email):
            missing = "your name and a phone number or email address"
        elif not name:
            missing = "your full name"
        else:
            missing = "a phone number or email address"
        return {
            "status": "need_contact",
            "message": (
                f"I would love to schedule that demo for you. "
                f"Before I book the slot, could you please share {missing}?"
            ),
        }

    time_candidate = (preferred_time_text or "").strip() or (user_text or "").strip()
    parsed_dt = parse_preferred_time(time_candidate, now=now)
    if parsed_dt is None:
        slots = get_available_slots(now=now, db_path=db_path)[:3]
        if not slots:
            return {
                "status": "no_slots",
                "message": "I do not see any open slots right now, so I will have a specialist follow up with you.",
            }
        labels = ", ".join(slot["label"] for slot in slots)
        return {
            "status": "need_time",
            "message": f"I can book your demo at {labels}. Which day and time works best for you?",
            "alternatives": slots,
        }

    result = book_slot(
        lead_id=lead_id,
        call_id=call_id,
        slot_start=parsed_dt,
        attendee_name=name,
        attendee_email=email,
        now=now,
        db_path=db_path,
    )
    if result["booked"]:
        return {
            "status": "booked",
            "message": (
                f"Great, I have booked your product demo for {result['label']}. "
                "You can also download the calendar invite right here."
            ),
            "booking": result,
        }

    alternatives = result.get("alternatives") or []
    if alternatives:
        alt_labels = "; ".join(slot["label"] for slot in alternatives)
        message = (
            f"That time is not available, but the nearest open slots are {alt_labels}. "
            "Would one of those work for you?"
        )
    else:
        message = "That slot is unavailable and no upcoming slots were found."

    return {
        "status": "conflict",
        "message": message,
        "alternatives": alternatives,
    }
