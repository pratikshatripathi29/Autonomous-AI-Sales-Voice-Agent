"""Streamlit UI entrypoint for the Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from core.analytics import build_score_gauge, build_sentiment_trend_figure
from core.config import COMPANY_NAME, KNOWLEDGE_DIR, configure_logging, get_config_status
from core.db import init_db
from core.leads import get_score_breakdown, score_lead, sync_lead_to_db
from core.llm import generate_turn, rewrite_search_query
from core.memory import (
    init_conversation_memory,
    record_message,
    update_session_lead_state,
)
from core.rag import build_index, format_context, retrieve
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


def _format_lead_status_badge(status: str) -> str:
    """Return a prominent colored Hot/Warm/Cold badge for the live lead panel."""
    if status == "Hot":
        return ":red-background[HOT LEAD]"
    if status == "Warm":
        return ":orange-background[WARM LEAD]"
    return ":blue-background[COLD LEAD]"


def _handle_knowledge_sidebar() -> None:
    """Render sidebar controls to upload knowledge docs and rebuild the FAISS index."""
    st.sidebar.divider()
    st.sidebar.caption("Knowledge Base (RAG)")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Knowledge Docs (.md, .txt)",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        processed_sigs = st.session_state.setdefault("uploaded_doc_sigs", set())
        saved_names: list[str] = []
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            raw_bytes = uploaded.getvalue()
            sig = f"{uploaded.name}:{hashlib.sha256(raw_bytes).hexdigest()}"
            if sig in processed_sigs:
                continue
            # Use Path(name).name to avoid path traversal from uploaded filenames.
            safe_name = Path(uploaded.name).name
            target_path = KNOWLEDGE_DIR / safe_name
            target_path.write_bytes(raw_bytes)
            processed_sigs.add(sig)
            saved_names.append(safe_name)

        if saved_names:
            with st.sidebar:
                with st.spinner("Indexing uploaded docs..."):
                    built = build_index()
            if built is not None:
                st.sidebar.success(f"Added & indexed: {', '.join(saved_names)}")
            else:
                st.sidebar.warning("Files saved, but index rebuild failed. Check logs.")

    if st.sidebar.button("Rebuild knowledge base", use_container_width=True):
        with st.sidebar:
            with st.spinner("Rebuilding FAISS index..."):
                built = build_index()
        if built is not None:
            st.sidebar.success("Knowledge base rebuilt.")
        else:
            st.sidebar.warning("Could not rebuild index (missing docs or GEMINI_API_KEY).")


def _process_voice_turn(audio_bytes: bytes) -> bool:
    """Run STT -> RAG -> LLM -> Lead Scoring/SQLite -> TTS for new audio."""
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
    history = st.session_state.get("llm_history", [])
    last_intent = st.session_state.get("last_intent")

    with st.spinner("Searching knowledge base..."):
        search_query = rewrite_search_query(
            user_text=user_text,
            history=history,
            last_intent=last_intent,
        )
        hits = retrieve(search_query, k=3)
        rag_context = format_context(hits)

    with st.spinner("Thinking..."):
        turn = generate_turn(
            user_text=user_text,
            history=history,
            rag_context=rag_context,
            lead_state=st.session_state.get("lead_state", {}),
            detected_lang=detected_lang,
            memory_summary=st.session_state.get("memory_summary", ""),
            last_intent=last_intent,
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
    updated_lead = update_session_lead_state(
        st.session_state,
        turn.get("extracted"),
        wants_booking=bool(turn.get("wants_booking")),
    )

    score, status = score_lead(
        updated_lead,
        sentiment_history=st.session_state.get("sentiment_history", []),
        intents=st.session_state.get("intents", []),
    )
    st.session_state["lead_score"] = score
    st.session_state["lead_status"] = status
    st.session_state["lead_breakdown"] = get_score_breakdown(
        updated_lead,
        sentiment_history=st.session_state.get("sentiment_history", []),
        intents=st.session_state.get("intents", []),
    )
    sync_lead_to_db(
        st.session_state,
        score=score,
        status=status,
        language=detected_lang,
    )

    reply_text = turn["reply"]
    reply_lang = turn.get("language") or detected_lang
    sources_payload = [
        {"source": source, "score": score_val, "text": chunk_text}
        for chunk_text, source, score_val in hits
    ]
    record_message(
        st.session_state,
        role="assistant",
        text=reply_text,
        language=reply_lang,
        sources=sources_payload,
    )

    with st.spinner("Synthesizing voice reply..."):
        tts_bytes = speak(reply_text, reply_lang)

    st.session_state["last_tts_audio"] = tts_bytes
    st.session_state["last_language"] = reply_lang
    return True


def _render_transcript() -> None:
    """Render the conversation transcript with sentiment badges and RAG sources."""
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
                sources = msg.get("sources") or []
                with st.expander("Sources used"):
                    if not sources:
                        st.write("No knowledge base sources matched this turn.")
                    else:
                        for item in sources:
                            st.markdown(
                                f"**{item['source']}** (distance: `{item['score']:.4f}`)"
                            )
                            st.caption(item["text"])


def _render_live_lead_panel() -> None:
    """Render the right-hand live lead card, Plotly score gauge, badge, and sentiment trend."""
    score = int(st.session_state.get("lead_score", 0))
    status = str(st.session_state.get("lead_status", "Cold"))
    lead_state = st.session_state.get("lead_state", {})
    breakdown = st.session_state.get("lead_breakdown", [])
    sentiment_history = st.session_state.get("sentiment_history", [])

    st.subheader("Live Lead Intelligence")
    st.markdown(_format_lead_status_badge(status))
    if st.session_state.get("lead_id"):
        st.caption(f"Saved Lead ID: `{st.session_state['lead_id']}`")

    st.plotly_chart(build_score_gauge(score, status), use_container_width=True)

    with st.expander("Score Breakdown", expanded=True):
        if not breakdown:
            st.caption("No qualification signals detected yet.")
        else:
            for reason in breakdown:
                st.write(f"- {reason}")

    with st.container(border=True):
        st.markdown("**Lead Card**")
        display_fields = [
            ("Name", lead_state.get("name")),
            ("Email", lead_state.get("email")),
            ("Phone", lead_state.get("phone")),
            ("Company", lead_state.get("company")),
            ("Need", lead_state.get("need")),
            ("Budget", lead_state.get("budget")),
            ("Timeline", lead_state.get("timeline")),
        ]
        for label, value in display_fields:
            st.write(f"**{label}:** {value or '—'}")

    if sentiment_history:
        st.plotly_chart(
            build_sentiment_trend_figure(sentiment_history),
            use_container_width=True,
        )
    else:
        st.caption("Sentiment trend will appear after your first spoken turn.")

    memory_summary = st.session_state.get("memory_summary", "")
    if memory_summary:
        st.subheader("Rolling Memory Summary")
        st.write(memory_summary)


def render_voice_agent_page() -> None:
    """Render the Voice Agent page with recording, transcript, TTS playback, and live lead panel."""
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

    chat_col, state_col = st.columns([3, 2])
    with chat_col:
        st.subheader("Conversation Transcript")
        _render_transcript()

    with state_col:
        _render_live_lead_panel()


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

    _handle_knowledge_sidebar()

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
