"""Streamlit UI entrypoint for the Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from core.analytics import (
    aggregate_objections_and_pain_points,
    build_calls_per_day_chart,
    build_conversion_funnel,
    build_language_distribution_chart,
    build_lead_status_pie,
    build_score_gauge,
    build_sentiment_over_time_chart,
    build_sentiment_trend_figure,
    build_top_issues_chart,
    compute_kpis,
    generate_business_insights,
    load_analytics_data,
)
from core.booking import generate_ics, process_booking_request
from core.config import (
    COMPANY_NAME,
    CRM_MODE,
    KNOWLEDGE_DIR,
    configure_logging,
    get_config_status,
)
from core.crm import sync_lead
from core.db import (
    get_crm_sync_logs,
    init_db,
    list_appointments,
    list_leads,
)
from core.leads import get_score_breakdown, score_lead, sync_lead_to_db
from core.llm import generate_turn, rewrite_search_query
from core.memory import (
    init_conversation_memory,
    record_message,
    update_session_lead_state,
)
from core.rag import build_index, format_context, retrieve
from core.stt import transcribe
from core.summary import summarize_call
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

    if st.sidebar.button("Rebuild knowledge base", width="stretch"):
        with st.sidebar:
            with st.spinner("Rebuilding FAISS index..."):
                built = build_index()
        if built is not None:
            st.sidebar.success("Knowledge base rebuilt.")
        else:
            st.sidebar.warning("Could not rebuild index (missing docs or GEMINI_API_KEY).")


def _process_voice_turn(audio_bytes: bytes) -> bool:
    """Run STT -> RAG -> LLM -> Lead Scoring -> Booking -> CRM Sync -> TTS for new audio."""
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
    lead_id = sync_lead_to_db(
        st.session_state,
        score=score,
        status=status,
        language=detected_lang,
    )

    reply_text = turn["reply"]
    reply_lang = turn.get("language") or detected_lang

    # Execute slot validation/booking when the customer wants to schedule a demo.
    if turn.get("wants_booking") or (
        turn.get("preferred_time") and st.session_state.get("lead_state", {}).get("wants_booking")
    ):
        booking_outcome = process_booking_request(
            lead_id=lead_id,
            call_id=st.session_state.get("call_id"),
            lead_state=updated_lead,
            preferred_time_text=turn.get("preferred_time"),
            user_text=user_text,
        )
        reply_text = booking_outcome["message"]
        if booking_outcome["status"] == "booked":
            booking_info = booking_outcome["booking"]
            st.session_state["last_booking"] = booking_info
            if lead_id is not None:
                sync_lead(
                    {
                        "id": lead_id,
                        "score": score,
                        "status": status,
                        "appointment": {
                            "slot_start": booking_info["slot_start"],
                            "slot_end": booking_info["slot_end"],
                            "status": "booked",
                        },
                        "summary": st.session_state.get("memory_summary", ""),
                    },
                    mode=CRM_MODE,
                )

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


def _handle_end_call() -> None:
    """Trigger summary -> final lead score -> CRM sync -> store summary card."""
    call_id = st.session_state.get("call_id")
    if call_id is None:
        return
    if not st.session_state.get("transcript"):
        st.warning("Record at least one conversation turn before ending the call.")
        return

    with st.spinner("Generating post-call summary..."):
        summary_data = summarize_call(int(call_id))

    lead_state = st.session_state.get("lead_state", {})
    sentiment_history = st.session_state.get("sentiment_history", [])
    intents = st.session_state.get("intents", [])

    final_score, final_status = score_lead(
        lead_state,
        sentiment_history=sentiment_history,
        intents=intents,
    )
    st.session_state["lead_score"] = final_score
    st.session_state["lead_status"] = final_status
    st.session_state["lead_breakdown"] = get_score_breakdown(
        lead_state,
        sentiment_history=sentiment_history,
        intents=intents,
    )

    lead_id = sync_lead_to_db(
        st.session_state,
        score=final_score,
        status=final_status,
        language=st.session_state.get("last_language", "en"),
    )

    crm_status = "skipped (no contact details captured)"
    if lead_id is not None:
        sync_result = sync_lead(
            {
                "id": int(lead_id),
                "score": final_score,
                "status": final_status,
                "summary": summary_data["summary"],
            },
            mode=CRM_MODE,
        )
        crm_status = sync_result["status"]

    st.session_state["last_call_summary"] = {
        **summary_data,
        "final_score": final_score,
        "final_status": final_status,
        "crm_status": crm_status,
    }


def _render_post_call_summary_card(summary_data: dict) -> None:
    """Render the structured post-call summary card on the Voice Agent page."""
    with st.container(border=True):
        st.subheader("Post-Call Summary Card")
        col1, col2, col3 = st.columns(3)
        col1.metric("Outcome", str(summary_data.get("outcome", "follow_up")).title())
        col2.metric(
            "Overall Sentiment",
            str(summary_data.get("overall_sentiment", "neutral")).title(),
        )
        col3.metric(
            "Final Lead Score",
            f"{summary_data.get('final_score', 0)} ({summary_data.get('final_status', 'Cold')})",
        )

        st.markdown(f"**Summary:** {summary_data.get('summary', '')}")
        st.markdown(f"**Recommended Action:** {summary_data.get('recommended_action', '')}")
        st.caption(f"CRM Sync: `{summary_data.get('crm_status', 'not synced')}`")

        list_col1, list_col2 = st.columns(2)
        with list_col1:
            st.markdown("**Key Points**")
            for item in summary_data.get("key_points") or ["None"]:
                st.write(f"- {item}")
            st.markdown("**Customer Pain Points**")
            for item in summary_data.get("customer_pain_points") or ["None"]:
                st.write(f"- {item}")
        with list_col2:
            st.markdown("**Objections**")
            for item in summary_data.get("objections") or ["None"]:
                st.write(f"- {item}")
            st.markdown("**Next Steps**")
            for item in summary_data.get("next_steps") or ["None"]:
                st.write(f"- {item}")


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
    last_booking = st.session_state.get("last_booking")

    st.subheader("Live Lead Intelligence")
    st.markdown(_format_lead_status_badge(status))
    if st.session_state.get("lead_id"):
        st.caption(f"Saved Lead ID: `{st.session_state['lead_id']}`")

    st.plotly_chart(build_score_gauge(score, status), width="stretch")

    if last_booking:
        st.success(f"Demo Booked: {last_booking['label']}")
        st.download_button(
            "Download Calendar Invite (.ics)",
            data=last_booking["ics"],
            file_name=f"demo_{last_booking['appointment_id']}.ics",
            mime="text/calendar",
            width="stretch",
        )

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
            width="stretch",
        )
    else:
        st.caption("Sentiment trend will appear after your first spoken turn.")

    memory_summary = st.session_state.get("memory_summary", "")
    if memory_summary:
        st.subheader("Rolling Memory Summary")
        st.write(memory_summary)


def render_voice_agent_page() -> None:
    """Render the Voice Agent page with recording, End Call summary, transcript, and live lead panel."""
    init_conversation_memory(st.session_state)

    header_col, end_col, new_col = st.columns([3, 1, 1])
    with header_col:
        st.header(f"{COMPANY_NAME} — Voice Agent")
        st.caption(f"Active Call ID: `{st.session_state.get('call_id')}`")
    with end_col:
        if st.button("End Call", width="stretch"):
            _handle_end_call()
    with new_col:
        if st.button("New Call", width="stretch"):
            init_conversation_memory(st.session_state, force_reset=True)
            st.session_state.pop("last_audio_hash", None)
            st.session_state.pop("last_stt_empty", None)
            # Rotate the st.audio_input key so stale recorded audio does not replay on rerun.
            st.session_state["audio_widget_nonce"] = (
                int(st.session_state.get("audio_widget_nonce", 0)) + 1
            )
            st.rerun()

    audio_widget_key = f"voice_audio_input_{st.session_state.get('audio_widget_nonce', 0)}"
    audio_file = st.audio_input("Record your message", key=audio_widget_key)
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

    summary_card = st.session_state.get("last_call_summary")
    if summary_card:
        _render_post_call_summary_card(summary_card)

    chat_col, state_col = st.columns([3, 2])
    with chat_col:
        st.subheader("Conversation Transcript")
        _render_transcript()

    with state_col:
        _render_live_lead_panel()


def render_leads_appointments_page() -> None:
    """Render interactive Leads, Appointments, and CRM Sync Log tables with filters and exports."""
    st.header("Leads & Appointments")

    leads_rows = list_leads()
    appt_rows = list_appointments()
    crm_rows = get_crm_sync_logs()

    leads_df = pd.DataFrame(leads_rows)
    appts_df = pd.DataFrame(appt_rows)
    crm_df = pd.DataFrame(crm_rows)

    st.subheader("Leads")
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        status_filter = st.selectbox("Filter by Status", options=["All", "Hot", "Warm", "Cold"])
    with filter_col2:
        search_text = st.text_input("Search Name, Email, or Company", placeholder="Type to filter...")

    filtered_leads_df = leads_df.copy()
    if not filtered_leads_df.empty:
        if status_filter != "All":
            filtered_leads_df = filtered_leads_df[filtered_leads_df["status"] == status_filter]
        if search_text.strip():
            term = search_text.strip().lower()
            mask = (
                filtered_leads_df["name"].fillna("").str.lower().str.contains(term)
                | filtered_leads_df["email"].fillna("").str.lower().str.contains(term)
                | filtered_leads_df["company"].fillna("").str.lower().str.contains(term)
            )
            filtered_leads_df = filtered_leads_df[mask]

    st.dataframe(filtered_leads_df, width="stretch")
    if not filtered_leads_df.empty:
        st.download_button(
            "Download Leads CSV",
            data=filtered_leads_df.to_csv(index=False).encode("utf-8"),
            file_name="leads_export.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Appointments")
    appt_status_filter = st.selectbox(
        "Filter Appointments by Status",
        options=["All", "booked", "cancelled"],
    )
    filtered_appts_df = appts_df.copy()
    if not filtered_appts_df.empty and appt_status_filter != "All":
        filtered_appts_df = filtered_appts_df[filtered_appts_df["status"] == appt_status_filter]

    st.dataframe(filtered_appts_df, width="stretch")
    if not filtered_appts_df.empty:
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download Appointments CSV",
                data=filtered_appts_df.to_csv(index=False).encode("utf-8"),
                file_name="appointments_export.csv",
                mime="text/csv",
            )
        with dl_col2:
            first_appt = filtered_appts_df.iloc[0]
            ics_data = generate_ics(
                slot_start=str(first_appt["slot_start"]),
                slot_end=str(first_appt["slot_end"]),
                uid=f"appt-{first_appt['id']}@novacloud.ai",
            )
            st.download_button(
                f"Download Latest .ics (Appt #{first_appt['id']})",
                data=ics_data,
                file_name=f"appointment_{first_appt['id']}.ics",
                mime="text/calendar",
            )

    st.divider()
    st.subheader("CRM Sync Log")
    if not leads_df.empty:
        sync_col1, sync_col2 = st.columns([2, 1])
        with sync_col1:
            lead_options = {
                f"#{row['id']} — {row['name']} ({row['status']})": int(row["id"])
                for _, row in leads_df.iterrows()
            }
            selected_label = st.selectbox("Select Lead for Manual CRM Sync", options=list(lead_options.keys()))
        with sync_col2:
            st.write("")
            st.write("")
            if st.button("Sync Lead Now", width="stretch"):
                result = sync_lead({"id": lead_options[selected_label]}, mode=CRM_MODE)
                if result["ok"]:
                    st.success(f"Synced ({result['status']})")
                    st.rerun()
                else:
                    st.warning(f"Sync status: {result['status']}")

    st.dataframe(crm_df, width="stretch")
    if not crm_df.empty:
        st.download_button(
            "Download CRM Sync Log CSV",
            data=crm_df.to_csv(index=False).encode("utf-8"),
            file_name="crm_sync_log.csv",
            mime="text/csv",
        )


def render_analytics_page() -> None:
    """Render the Plotly executive analytics dashboard and AI business insights."""
    st.header("Sales Analytics & Business Insights")

    data = load_analytics_data()
    kpis = compute_kpis(data)
    issues = aggregate_objections_and_pain_points(data["calls"])

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Calls", kpis["total_calls"])
    kpi_cols[1].metric("Total Leads", kpis["total_leads"])
    kpi_cols[2].metric("Hot Leads %", f"{kpis['hot_leads_pct']:.1f}%")
    kpi_cols[3].metric("Booked Demos", kpis["appointments_booked"])
    kpi_cols[4].metric("Conversion Rate", f"{kpis['conversion_rate']:.1f}%")
    kpi_cols[5].metric("Avg Sentiment", f"{kpis['avg_sentiment']:+.2f}")

    st.divider()
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(build_conversion_funnel(kpis), width="stretch")
    with row1_col2:
        st.plotly_chart(build_lead_status_pie(data["leads"]), width="stretch")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(
            build_sentiment_over_time_chart(data["calls"], data["messages"]),
            width="stretch",
        )
    with row2_col2:
        st.plotly_chart(build_calls_per_day_chart(data["calls"]), width="stretch")

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(
            build_language_distribution_chart(data["messages"], data["leads"]),
            width="stretch",
        )
    with row3_col2:
        st.plotly_chart(build_top_issues_chart(issues), width="stretch")

    st.divider()
    st.subheader("AI Business Insights")
    if st.button("AI business insights", type="primary"):
        with st.spinner("Analyzing pipeline metrics with Gemini..."):
            st.session_state["ai_business_insights"] = generate_business_insights(data)

    insights = st.session_state.get("ai_business_insights")
    if insights:
        for idx, insight in enumerate(insights, start=1):
            st.info(f"**{idx}.** {insight}")


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
    st.sidebar.write(f"CRM Mode: `{CRM_MODE}`")

    if page == "Voice Agent":
        render_voice_agent_page()
    elif page == "Leads & Appointments":
        render_leads_appointments_page()
    else:
        render_analytics_page()


if __name__ == "__main__":
    main()
