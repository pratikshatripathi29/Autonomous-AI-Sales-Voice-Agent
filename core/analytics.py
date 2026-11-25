"""Plotly chart builders, KPI calculations, summary aggregation, and AI business insights."""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
import plotly.graph_objects as go

from core.config import GEMINI_API_KEY
from core.db import list_all_messages, list_appointments, list_calls, list_leads
from core.llm import _create_chat_client, _extract_response_text, _strip_code_fences
from core.summary import parse_call_summary
from core.tts import normalize_lang_code

logger = logging.getLogger(__name__)


def _empty_figure(title: str, message: str = "No data yet") -> go.Figure:
    """Return a clean placeholder Plotly figure when no records exist."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#6b7280"},
    )
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


# --- Live Voice Page Widgets ---


def build_score_gauge(score: int, status: str) -> go.Figure:
    """Build a Plotly indicator gauge visualizing the 0-100 lead score."""
    bar_color = "#dc2626" if status == "Hot" else ("#f59e0b" if status == "Warm" else "#2563eb")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            title={"text": f"Lead Score ({status})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 40], "color": "#dbeafe"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def build_sentiment_trend_figure(sentiment_history: list[float]) -> go.Figure:
    """Build a Plotly line chart tracking turn-by-turn customer sentiment."""
    turns = list(range(1, len(sentiment_history) + 1))
    fig = go.Figure(
        go.Scatter(
            x=turns,
            y=sentiment_history,
            mode="lines+markers",
            name="Sentiment",
            line={"color": "#10b981", "width": 2},
            marker={"size": 8},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Sentiment Trend",
        xaxis_title="Turn",
        yaxis_title="Score (-1 to +1)",
        yaxis=dict(range=[-1.1, 1.1]),
        xaxis=dict(dtick=1),
        height=230,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- Analytics Aggregation & Dashboard Charts ---


def load_analytics_data(db_path: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load calls, leads, appointments, and messages from SQLite for dashboarding."""
    return {
        "calls": list_calls(db_path=db_path),
        "leads": list_leads(db_path=db_path),
        "appointments": list_appointments(db_path=db_path),
        "messages": list_all_messages(db_path=db_path),
    }


def compute_kpis(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute top-level sales KPIs from loaded SQLite records."""
    calls = data.get("calls") or []
    leads = data.get("leads") or []
    appointments = data.get("appointments") or []
    messages = data.get("messages") or []

    total_calls = len(calls)
    total_leads = len(leads)
    hot_leads = sum(1 for lead in leads if str(lead.get("status", "")).lower() == "hot")
    hot_leads_pct = (hot_leads / total_leads * 100.0) if total_leads else 0.0

    booked_appts = sum(
        1 for appt in appointments if str(appt.get("status", "")).lower() == "booked"
    )
    conversion_rate = (booked_appts / total_calls * 100.0) if total_calls else 0.0

    call_sentiments = [
        float(call["sentiment_avg"])
        for call in calls
        if call.get("sentiment_avg") is not None
    ]
    if not call_sentiments:
        call_sentiments = [
            float(msg["sentiment"])
            for msg in messages
            if msg.get("role") == "user" and msg.get("sentiment") is not None
        ]
    avg_sentiment = (
        sum(call_sentiments) / len(call_sentiments) if call_sentiments else 0.0
    )

    return {
        "total_calls": total_calls,
        "total_leads": total_leads,
        "hot_leads": hot_leads,
        "hot_leads_pct": round(hot_leads_pct, 1),
        "appointments_booked": booked_appts,
        "conversion_rate": round(conversion_rate, 1),
        "avg_sentiment": round(avg_sentiment, 2),
    }


def aggregate_objections_and_pain_points(
    calls: list[dict[str, Any]],
    top_n: int = 8,
) -> dict[str, list[tuple[str, int]]]:
    """Aggregate objection and pain-point frequencies from saved call summaries."""
    objections_counter: Counter[str] = Counter()
    pain_counter: Counter[str] = Counter()

    for call in calls:
        parsed = parse_call_summary(call.get("summary"))
        for obj in parsed.get("objections") or []:
            objections_counter[obj] += 1
        for pain in parsed.get("customer_pain_points") or []:
            pain_counter[pain] += 1

    return {
        "objections": objections_counter.most_common(top_n),
        "pain_points": pain_counter.most_common(top_n),
    }


def build_conversion_funnel(kpis: dict[str, Any]) -> go.Figure:
    """Build a Plotly funnel chart from Calls -> Leads -> Hot Leads -> Booked."""
    stages = ["Calls", "Leads Captured", "Hot Leads", "Appointments Booked"]
    values = [
        kpis.get("total_calls", 0),
        kpis.get("total_leads", 0),
        kpis.get("hot_leads", 0),
        kpis.get("appointments_booked", 0),
    ]
    if sum(values) == 0:
        return _empty_figure("Sales Conversion Funnel")

    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker={"color": ["#3b82f6", "#6366f1", "#f59e0b", "#10b981"]},
        )
    )
    fig.update_layout(title="Sales Conversion Funnel", height=340, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def build_lead_status_pie(leads: list[dict[str, Any]]) -> go.Figure:
    """Build a Plotly donut chart showing Hot / Warm / Cold lead distribution."""
    if not leads:
        return _empty_figure("Lead Status Breakdown")

    counts = Counter(str(lead.get("status") or "Cold") for lead in leads)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    color_map = {"Hot": "#dc2626", "Warm": "#f59e0b", "Cold": "#2563eb"}
    colors = [color_map.get(label, "#6b7280") for label in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker={"colors": colors},
        )
    )
    fig.update_layout(title="Lead Status Distribution", height=340, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def build_sentiment_over_time_chart(
    calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> go.Figure:
    """Build a Plotly line chart of sentiment over time across calls or user messages."""
    points: list[tuple[str, float]] = []
    for call in reversed(calls):
        if call.get("sentiment_avg") is not None and call.get("started_at"):
            points.append((str(call["started_at"])[:19].replace("T", " "), float(call["sentiment_avg"])))

    if not points:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("sentiment") is not None and msg.get("created_at"):
                points.append((str(msg["created_at"])[:19].replace("T", " "), float(msg["sentiment"])))

    if not points:
        return _empty_figure("Sentiment Over Time")

    fig = go.Figure(
        go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode="lines+markers",
            line={"color": "#10b981", "width": 2},
            marker={"size": 7},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Sentiment (-1 to +1)",
        yaxis=dict(range=[-1.1, 1.1]),
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def build_calls_per_day_chart(calls: list[dict[str, Any]]) -> go.Figure:
    """Build a Plotly bar chart showing call volume grouped by day."""
    if not calls:
        return _empty_figure("Calls Per Day")

    day_counts = Counter(
        str(call["started_at"])[:10] for call in calls if call.get("started_at")
    )
    days = sorted(day_counts.keys())
    counts = [day_counts[d] for d in days]

    fig = go.Figure(
        go.Bar(x=days, y=counts, marker_color="#3b82f6", text=counts, textposition="auto")
    )
    fig.update_layout(
        title="Calls Per Day",
        xaxis_title="Date",
        yaxis_title="Calls",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def build_language_distribution_chart(
    messages: list[dict[str, Any]],
    leads: list[dict[str, Any]],
) -> go.Figure:
    """Build a Plotly bar chart of detected languages across user utterances or leads."""
    user_langs = [
        normalize_lang_code(str(msg.get("language") or "en")).upper()
        for msg in messages
        if msg.get("role") == "user"
    ]
    if not user_langs:
        user_langs = [
            normalize_lang_code(str(lead.get("language") or "en")).upper()
            for lead in leads
        ]

    if not user_langs:
        return _empty_figure("Language Distribution")

    counts = Counter(user_langs)
    langs = list(counts.keys())
    vals = [counts[lang] for lang in langs]

    fig = go.Figure(
        go.Bar(x=langs, y=vals, marker_color="#8b5cf6", text=vals, textposition="auto")
    )
    fig.update_layout(
        title="Language Distribution",
        xaxis_title="Language Code",
        yaxis_title="Count",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def build_top_issues_chart(aggregated: dict[str, list[tuple[str, int]]]) -> go.Figure:
    """Build a Plotly bar chart comparing top customer objections and pain points."""
    objections = aggregated.get("objections") or []
    pain_points = aggregated.get("pain_points") or []
    if not objections and not pain_points:
        return _empty_figure("Top Objections & Pain Points", "End calls to populate summary insights")

    labels: list[str] = []
    counts: list[int] = []
    categories: list[str] = []

    for text, count in objections[:5]:
        labels.append(f"Obj: {text}")
        counts.append(count)
        categories.append("Objection")
    for text, count in pain_points[:5]:
        labels.append(f"Pain: {text}")
        counts.append(count)
        categories.append("Pain Point")

    colors = ["#ef4444" if cat == "Objection" else "#f59e0b" for cat in categories]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Top Objections & Pain Points",
        xaxis_title="Mentions",
        yaxis={"autorange": "reversed"},
        height=360,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def generate_business_insights(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Send aggregate sales metrics to Gemini and return 3-5 actionable business insights."""
    kpis = compute_kpis(data)
    issues = aggregate_objections_and_pain_points(data.get("calls") or [])
    leads = data.get("leads") or []
    status_counts = dict(Counter(str(lead.get("status") or "Cold") for lead in leads))
    lang_counts = dict(
        Counter(
            normalize_lang_code(str(msg.get("language") or "en"))
            for msg in (data.get("messages") or [])
            if msg.get("role") == "user"
        )
    )

    fallback = [
        f"Handled {kpis['total_calls']} calls capturing {kpis['total_leads']} leads ({kpis['hot_leads_pct']}% Hot).",
        f"Booked {kpis['appointments_booked']} demos ({kpis['conversion_rate']}% call-to-booking conversion).",
        f"Average customer sentiment is {kpis['avg_sentiment']:+.2f}; prioritize follow-ups on Warm/Hot leads.",
    ]

    if kpis["total_calls"] == 0 and kpis["total_leads"] == 0:
        return ["No call or lead data recorded yet. Run a voice conversation to generate AI insights."]

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY missing; returning deterministic business insights.")
        return fallback

    stats_payload = {
        "kpis": kpis,
        "lead_status_breakdown": status_counts,
        "user_language_counts": lang_counts,
        "top_objections": issues["objections"],
        "top_pain_points": issues["pain_points"],
    }
    prompt = (
        "You are a VP of Sales Operations. Given these aggregate AI voice agent metrics, "
        "return ONLY valid JSON of the form {\"insights\": [string, ...]} with 3 to 5 concise, "
        "actionable business insights referencing concrete numbers from the stats.\n\n"
        f"METRICS:\n{json.dumps(stats_payload, ensure_ascii=False)}"
    )

    try:
        client = _create_chat_client(json_mode=True, temperature=0.3)
        response = client.invoke([HumanMessage(content=prompt)])
        raw_text = _strip_code_fences(_extract_response_text(response))
        parsed = json.loads(raw_text)
        insights = parsed.get("insights") if isinstance(parsed, dict) else None
        if isinstance(insights, list):
            cleaned = [str(item).strip() for item in insights if str(item).strip()]
            if cleaned:
                return cleaned[:5]
    except Exception:
        logger.exception("Gemini generate_business_insights failed; using fallback insights.")

    return fallback
