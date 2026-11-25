"""Plotly chart builders for live call widgets and post-call analytics."""

from __future__ import annotations

import plotly.graph_objects as go


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
