"""Conversation memory management with rolling LLM summarization and SQLite persistence."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from core.config import MEMORY_WINDOW_SIZE
from core.db import add_message, create_call
from core.llm import EXTRACTED_KEYS, summarize_messages


def create_empty_lead_state() -> dict[str, Any]:
    """Return an empty lead qualification state dictionary."""
    return {key: None for key in EXTRACTED_KEYS}


def merge_lead_state(
    current_state: dict[str, Any] | None,
    extracted: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge newly extracted non-empty lead fields into the existing lead_state."""
    merged = create_empty_lead_state()
    if current_state:
        for key in EXTRACTED_KEYS:
            if current_state.get(key):
                merged[key] = current_state[key]

    if extracted:
        for key in EXTRACTED_KEYS:
            value = extracted.get(key)
            # Only overwrite when the LLM extracted a concrete value so prior facts are preserved.
            if isinstance(value, str):
                value = value.strip() or None
            if value is not None:
                merged[key] = value
    return merged


def init_conversation_memory(
    state: MutableMapping[str, Any],
    force_reset: bool = False,
    db_path: Path | str | None = None,
) -> int:
    """Initialize conversation history, summary, lead_state, and active call_id in state."""
    if force_reset or "call_id" not in state or state["call_id"] is None:
        state["call_id"] = create_call(lead_id=None, db_path=db_path)
        state["llm_history"] = []
        state["transcript"] = []
        state["memory_summary"] = ""
        state["lead_state"] = create_empty_lead_state()
        state["last_tts_audio"] = b""
        state["last_language"] = "en"
        state["last_intent"] = None
    else:
        state.setdefault("llm_history", [])
        state.setdefault("transcript", [])
        state.setdefault("memory_summary", "")
        state.setdefault("lead_state", create_empty_lead_state())
        state.setdefault("last_intent", None)
    return int(state["call_id"])


def record_message(
    state: MutableMapping[str, Any],
    role: str,
    text: str,
    language: str = "en",
    sentiment_label: str | None = None,
    sentiment_score: float | None = None,
    intent: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    max_window: int = MEMORY_WINDOW_SIZE,
    db_path: Path | str | None = None,
) -> int:
    """Persist a message to SQLite, append to UI transcript, and roll the LLM memory window."""
    call_id = init_conversation_memory(state, force_reset=False, db_path=db_path)

    msg_id = add_message(
        call_id=call_id,
        role=role,
        text=text,
        language=language,
        sentiment=sentiment_score,
        db_path=db_path,
    )

    entry: dict[str, Any] = {
        "id": msg_id,
        "role": role,
        "text": text,
        "language": language,
        "sentiment": sentiment_label or "neutral",
        "sentiment_score": sentiment_score,
        "intent": intent,
        "sources": sources or [],
    }
    if intent:
        state["last_intent"] = intent
    state["transcript"].append(entry)
    state["llm_history"].append({"role": role, "text": text})

    # Keep only the latest N messages in full for the LLM prompt and compress older turns.
    if len(state["llm_history"]) > max_window:
        overflow_count = len(state["llm_history"]) - max_window
        older_messages = state["llm_history"][:overflow_count]
        state["llm_history"] = state["llm_history"][overflow_count:]
        state["memory_summary"] = summarize_messages(
            older_messages,
            existing_summary=state.get("memory_summary", ""),
        )

    return msg_id


def update_session_lead_state(
    state: MutableMapping[str, Any],
    extracted: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge extracted turn entities into state['lead_state'] and return the updated dict."""
    updated = merge_lead_state(state.get("lead_state"), extracted)
    state["lead_state"] = updated
    return updated
