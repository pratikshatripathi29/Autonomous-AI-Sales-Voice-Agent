"""Streamlit UI entrypoint for the Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import hashlib

import streamlit as st

from core.config import COMPANY_NAME, configure_logging, get_config_status
from core.db import init_db
from core.llm import generate_turn
from core.memory import (
    init_conversation_memory,
    record_message,
    update_session_lead_state,
)
from core.stt import transcribe
from core.tts import normalize_lang_code, speak


def initialize_app() -> None:
    """Configure logging and initialize the SQLite database on app startup."""
    configure_logging()
    # Keep initialization in session_state so Streamlit reruns do not log DDL every click.
    if not st.session_state.get("db_initialized"):
        init_db()
        st.session_state["db_initialized"] = True


def _format_sentiment_badge(sentiment: str | None, score: float | None) -> str:
    """Return a colored Streamlit markdown badge for a message's sentiment."""
    label = (sentiment or "neutral").lower()
    score_text = f" ({score:+.2f})" if score is not None else ""
    if label == "positive":
        return f":green[Sentiment: Positive{score_text}]"
    if label == "negative":
        return f":red[Sentiment: Negative{score_text}]"
    return f":gray[Sentiment: Neutral{score_text}]"


def _process_voice_turn(audio_bytes: bytes) -> bool:
    """Run STT -> LLM -> memory/SQLite -> TTS for new audio, returning True if processed."""
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    # Skip STT/LLM/TTS API calls when Streamlit reruns with the exact same audio bytes.
    if audio_hash == st.session_state.get("last_audio_hash"):
        return False

    with st.spinner("Transcribing audio..."):
        stt_result = transcribe(audio_bytes)

    user_text = stt_result.get("text", "").strip()
    detected_lang = stt_result.get("language", "en")

    st.session_state["last_audio_hash"] = audio_hash
    if not user_text:
        st.session_state["last_stt_empty"] = True
        st.session_state["last_tts_audio"] = b""
        return True

    st.session_state["last_stt_empty"] = False
    with st.spinner("Thinking..."):
        turn = generate_turn(
            user_text=user_text,
            history=st.session_state.get("llm_history", []),
            rag_context="",
            lead_state=st.session_state.get("lead_state", {}),
            detected_lang=detected_lang,
            memory_summary=st.session_state.get("memory_summary", ""),
        )

    record_message(
        st.session_state,
        role="user",
        text=user_text,
        language=detected_lang,
        sentiment_label=turn["sentiment"],
        sentiment_score=turn["sentiment_score"],
        intent=turn["intent"],
    )
    update_session_lead_state(st.session_state, turn.get("extracted"))

    reply_text = turn["reply"]
    reply_lang = turn.get("language") or detected_lang
    record_message(
        st.session_state,
        role="assistant",
        text=reply_text,
        language=reply_lang,
    )

    with st.spinner("Synthesizing voice reply..."):
        tts_bytes = speak(reply_text, reply_lang)

    st.session_state["last_tts_audio"] = tts_bytes
    st.session_state["last_language"] = reply_lang
    return True


def _render_transcript() -> None:
    """Render the conversation transcript with sentiment badges and intent metadata."""
    transcript = st.session_state.get("transcript", [])
    if not transcript:
        st.info("Record a message above to start talking with the AI sales representative.")
        return

    for msg in transcript:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.write(msg.get("text", ""))
            if role == "user":
                badge = _format_sentiment_badge(
                    msg.get("sentiment"),
                    msg.get("sentiment_score"),
                )
                intent = msg.get("intent") or "other"
                lang = msg.get("language") or "en"
                st.caption(f"{badge} | Intent: `{intent}` | Lang: `{lang}`")
            else:
                lang = msg.get("language") or "en"
                st.caption(f"Lang: `{lang}` (TTS: `{normalize_lang_code(lang)}`)")


def render_voice_agent_page() -> None:
    """Render the Voice Agent page with recording, transcript, TTS playback, and lead state."""
    init_conversation_memory(st.session_state)

    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.header(f"{COMPANY_NAME} — Voice Agent")
        st.caption(f"Active Call ID: `{st.session_state.get('call_id')}`")
    with action_col:
        if st.button("New Call", use_container_width=True):
            init_conversation_memory(st.session_state, force_reset=True)
            st.session_state.pop("last_audio_hash", None)
            st.session_state.pop("last_stt_empty", None)
            st.rerun()

    audio_file = st.audio_input("Record your message")
    is_new_audio = False
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        if audio_bytes:
            is_new_audio = _process_voice_turn(audio_bytes)

    if st.session_state.get("last_stt_empty"):
        st.warning("Could not transcribe audio (empty speech or STT error). Check logs and GROQ_API_KEY.")

    tts_bytes = st.session_state.get("last_tts_audio", b"")
    if tts_bytes:
        # Autoplay only on the turn where new audio arrived; keep player visible on reruns.
        st.audio(tts_bytes, format="audio/mp3", autoplay=is_new_audio)

    chat_col, state_col = st.columns([2, 1])
    with chat_col:
        st.subheader("Conversation Transcript")
        _render_transcript()

    with state_col:
        st.subheader("Captured Lead State")
        st.json(st.session_state.get("lead_state", {}))
        memory_summary = st.session_state.get("memory_summary", "")
        if memory_summary:
            st.subheader("Rolling Memory Summary")
            st.write(memory_summary)


def render_leads_appointments_page() -> None:
    """Render the placeholder Leads & Appointments page."""
    st.header("Leads & Appointments")
    st.info("Lead qualification table, appointment calendar, and CRM sync logs will appear here.")


def render_analytics_page() -> None:
    """Render the placeholder Analytics page."""
    st.header("Analytics")
    st.info("Plotly charts for call outcomes, lead scores, and sentiment trends will appear here.")


def main() -> None:
    """Render the Streamlit navigation shell and route to the selected page."""
    st.set_page_config(
        page_title="Autonomous AI Sales Voice Agent",
        page_icon="🎙️",
        layout="wide",
    )
    initialize_app()

    st.sidebar.title("AI Sales Voice Agent")
    page = st.sidebar.radio(
        "Navigation",
        options=["Voice Agent", "Leads & Appointments", "Analytics"],
    )

    # Show configuration readiness indicators without exposing secret values.
    status = get_config_status()
    st.sidebar.divider()
    st.sidebar.caption("Environment Status")
    st.sidebar.write(f"Gemini Key: {'OK' if status['gemini_configured'] else 'Missing'}")
    st.sidebar.write(f"Groq Key: {'OK' if status['groq_configured'] else 'Missing'}")

    if page == "Voice Agent":
        render_voice_agent_page()
    elif page == "Leads & Appointments":
        render_leads_appointments_page()
    else:
        render_analytics_page()


if __name__ == "__main__":
    main()
