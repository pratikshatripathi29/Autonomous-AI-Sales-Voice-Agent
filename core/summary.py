"""Post-call structured summarization via Gemini and persistence to the calls table."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from core.config import GEMINI_API_KEY
from core.db import end_call, get_call_messages
from core.llm import _create_chat_client, _extract_response_text, _strip_code_fences

logger = logging.getLogger(__name__)

VALID_OUTCOMES: frozenset[str] = frozenset(
    {"booked", "qualified", "follow_up", "not_interested", "support", "other"}
)
VALID_OVERALL_SENTIMENTS: frozenset[str] = frozenset({"positive", "neutral", "negative"})


def _to_str_list(value: Any) -> list[str]:
    """Normalize a JSON field into a clean list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _default_call_summary(
    summary: str = "Call completed.",
    outcome: str = "follow_up",
    overall_sentiment: str = "neutral",
) -> dict[str, Any]:
    """Return a default structured call summary matching the Stage 7 schema."""
    return {
        "summary": summary,
        "key_points": [],
        "customer_pain_points": [],
        "objections": [],
        "next_steps": [],
        "outcome": outcome,
        "overall_sentiment": overall_sentiment,
        "recommended_action": "Follow up with the prospect by email or phone.",
    }


def _normalize_summary_payload(data: dict[str, Any], default_sentiment: str) -> dict[str, Any]:
    """Validate and coerce parsed JSON summary fields into the expected structure."""
    outcome = str(data.get("outcome") or "follow_up").strip().lower()
    if outcome not in VALID_OUTCOMES:
        outcome = "follow_up"

    overall_sentiment = str(data.get("overall_sentiment") or default_sentiment).strip().lower()
    if overall_sentiment not in VALID_OVERALL_SENTIMENTS:
        overall_sentiment = default_sentiment

    return {
        "summary": str(data.get("summary") or "").strip() or "Call completed.",
        "key_points": _to_str_list(data.get("key_points")),
        "customer_pain_points": _to_str_list(data.get("customer_pain_points")),
        "objections": _to_str_list(data.get("objections")),
        "next_steps": _to_str_list(data.get("next_steps")),
        "outcome": outcome,
        "overall_sentiment": overall_sentiment,
        "recommended_action": str(data.get("recommended_action") or "").strip()
        or "Review transcript and schedule follow-up.",
    }


def parse_call_summary(raw_summary: str | None) -> dict[str, Any]:
    """Parse a stored calls.summary string (JSON or legacy plain text) into a dict."""
    cleaned = (raw_summary or "").strip()
    if not cleaned:
        return _default_call_summary(summary="")

    if cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return _normalize_summary_payload(parsed, default_sentiment="neutral")
        except json.JSONDecodeError:
            pass

    return _default_call_summary(summary=cleaned)


def summarize_call(
    call_id: int,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load call messages, generate a structured Gemini JSON summary, and save to calls."""
    messages = get_call_messages(call_id, db_path=db_path)
    sentiments = [
        float(msg["sentiment"])
        for msg in messages
        if msg.get("role") == "user" and msg.get("sentiment") is not None
    ]
    sentiment_avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
    default_sentiment = (
        "positive"
        if sentiment_avg > 0.1
        else ("negative" if sentiment_avg < -0.1 else "neutral")
    )

    if not messages:
        empty_summary = _default_call_summary(
            summary="No conversation messages recorded for this call.",
            outcome="other",
            overall_sentiment=default_sentiment,
        )
        end_call(
            call_id,
            sentiment_avg=sentiment_avg,
            summary=json.dumps(empty_summary, ensure_ascii=False),
            outcome=empty_summary["outcome"],
            db_path=db_path,
        )
        return empty_summary

    transcript_text = "\n".join(
        f"{str(msg.get('role', 'user')).upper()}: {msg.get('text', '')}"
        for msg in messages
    )

    summary_data = _default_call_summary(
        summary=f"Call with {len(messages)} messages recorded.",
        overall_sentiment=default_sentiment,
    )

    if GEMINI_API_KEY:
        prompt = (
            "Analyze this sales call transcript and return ONLY valid JSON with these exact keys:\n"
            "{\n"
            '  "summary": string (2-3 sentence executive summary),\n'
            '  "key_points": list of strings,\n'
            '  "customer_pain_points": list of strings,\n'
            '  "objections": list of strings,\n'
            '  "next_steps": list of strings,\n'
            '  "outcome": "booked" | "qualified" | "follow_up" | "not_interested" | "support" | "other",\n'
            '  "overall_sentiment": "positive" | "neutral" | "negative",\n'
            '  "recommended_action": string\n'
            "}\n\n"
            f"TRANSCRIPT:\n{transcript_text}"
        )
        try:
            client = _create_chat_client(json_mode=True, temperature=0.2)
            response = client.invoke([HumanMessage(content=prompt)])
            raw_text = _strip_code_fences(_extract_response_text(response))
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                summary_data = _normalize_summary_payload(
                    parsed,
                    default_sentiment=default_sentiment,
                )
        except Exception:
            logger.exception("Gemini summarize_call failed for call_id=%s.", call_id)
    else:
        logger.warning("GEMINI_API_KEY missing; using deterministic fallback call summary.")

    # Persist the full JSON payload in calls.summary so analytics can aggregate pain points
    # and objections across calls without altering the SQLite schema.
    end_call(
        call_id,
        sentiment_avg=sentiment_avg,
        summary=json.dumps(summary_data, ensure_ascii=False),
        outcome=summary_data["outcome"],
        db_path=db_path,
    )
    return summary_data
