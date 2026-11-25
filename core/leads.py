"""Transparent rule-based lead scoring, breakdown explanations, and SQLite sync."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
import re
from typing import Any

from core.db import assign_call_lead, create_lead, update_lead

URGENT_TIMELINE_KEYWORDS: tuple[str, ...] = (
    "asap",
    "immediate",
    "immediately",
    "now",
    "today",
    "tomorrow",
    "right away",
    "this week",
    "next week",
    "this month",
    "next month",
    "urgent",
    "soon",
)


def _is_timeline_under_3_months(timeline_text: str | None) -> bool:
    """Return True if a timeline string indicates urgency under 3 months."""
    if not timeline_text:
        return False
    lowered = timeline_text.strip().lower()
    if any(keyword in lowered for keyword in URGENT_TIMELINE_KEYWORDS):
        return True

    month_match = re.search(r"(\d+(?:\.\d+)?)\s*month", lowered)
    if month_match and float(month_match.group(1)) < 3.0:
        return True

    week_match = re.search(r"(\d+(?:\.\d+)?)\s*week", lowered)
    if week_match and float(week_match.group(1)) < 13.0:
        return True

    day_match = re.search(r"(\d+(?:\.\d+)?)\s*day", lowered)
    if day_match and float(day_match.group(1)) < 90.0:
        return True

    return False


def _evaluate_rules(
    lead_state: dict[str, Any] | None,
    sentiment_history: list[float] | None,
    intents: list[str] | None,
) -> tuple[int, str, list[str]]:
    """Compute clamped score, Hot/Warm/Cold status, and human-readable rule reasons."""
    state = lead_state or {}
    sentiments = [float(s) for s in (sentiment_history or []) if s is not None]
    normalized_intents = {str(i).strip().lower() for i in (intents or []) if i}

    score = 0
    reasons: list[str] = []

    budget = str(state.get("budget") or "").strip()
    if budget:
        score += 20
        reasons.append(f"+20: Budget provided ({budget})")

    timeline = str(state.get("timeline") or "").strip()
    if _is_timeline_under_3_months(timeline):
        score += 20
        reasons.append(f"+20: Timeline under 3 months ({timeline})")

    company = str(state.get("company") or "").strip()
    if company:
        score += 15
        reasons.append(f"+15: Company / organization provided ({company})")

    email = str(state.get("email") or "").strip()
    phone = str(state.get("phone") or "").strip()
    if email or phone:
        score += 15
        reasons.append("+15: Contact details provided")

    if "pricing" in normalized_intents:
        score += 10
        reasons.append("+10: Asked about pricing")

    if bool(state.get("wants_booking")) or "booking" in normalized_intents:
        score += 20
        reasons.append("+20: Expressed interest in booking a demo")

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        # Penalize sustained negative sentiment without penalizing neutral (0.0) calls.
        if avg_sentiment < 0.0:
            score -= 15
            reasons.append(f"-15: Average sentiment is negative ({avg_sentiment:+.2f})")

    clamped_score = max(0, min(100, score))
    if clamped_score >= 70:
        status = "Hot"
    elif clamped_score >= 40:
        status = "Warm"
    else:
        status = "Cold"

    return clamped_score, status, reasons


def score_lead(
    lead_state: dict[str, Any] | None,
    sentiment_history: list[float] | None = None,
    intents: list[str] | None = None,
) -> tuple[int, str]:
    """Compute a 0-100 lead score and 'Hot', 'Warm', or 'Cold' classification."""
    score, status, _ = _evaluate_rules(lead_state, sentiment_history, intents)
    return score, status


def get_score_breakdown(
    lead_state: dict[str, Any] | None,
    sentiment_history: list[float] | None = None,
    intents: list[str] | None = None,
) -> list[str]:
    """Return a list of transparent scoring rule explanations for display."""
    _, _, reasons = _evaluate_rules(lead_state, sentiment_history, intents)
    return reasons


def sync_lead_to_db(
    state: MutableMapping[str, Any],
    score: int,
    status: str,
    language: str = "en",
    db_path: Path | str | None = None,
) -> int | None:
    """Create a lead on first captured name/phone/email, or update the existing lead row."""
    lead_state = state.get("lead_state") or {}
    name = str(lead_state.get("name") or "").strip()
    phone = str(lead_state.get("phone") or "").strip() or None
    email = str(lead_state.get("email") or "").strip() or None
    company = str(lead_state.get("company") or "").strip() or None

    existing_lead_id = state.get("lead_id")
    if existing_lead_id is not None:
        update_lead(
            int(existing_lead_id),
            name=name or "Unknown Lead",
            phone=phone,
            email=email,
            company=company,
            language=language or "en",
            score=score,
            status=status,
            db_path=db_path,
        )
        return int(existing_lead_id)

    # Wait until we have at least one identifying detail before creating a SQLite lead row.
    if not (name or phone or email):
        return None

    new_lead_id = create_lead(
        name=name or "Unknown Lead",
        phone=phone,
        email=email,
        company=company,
        language=language or "en",
        score=score,
        status=status,
        db_path=db_path,
    )
    state["lead_id"] = new_lead_id

    call_id = state.get("call_id")
    if call_id is not None:
        assign_call_lead(int(call_id), new_lead_id, db_path=db_path)

    return new_lead_id
