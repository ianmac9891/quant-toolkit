"""Semantic color palette and chart helpers shared across all pages."""

from __future__ import annotations

import plotly.graph_objects as go

# Semantic palette — matches config.toml primaryColor
PRIMARY   = "#4F8EF7"   # lines, bars, main series
BENCHMARK = "#E8A33D"   # comparison / benchmark series
POSITIVE  = "#3FB66B"   # gains, good outcomes
NEGATIVE  = "#E5564E"   # losses, warnings
NEUTRAL   = "#8A92A6"   # secondary / muted elements

# Opacity variants used in band fills (PRIMARY base)
PRIMARY_10 = "rgba(79,142,247,0.10)"
PRIMARY_18 = "rgba(79,142,247,0.18)"
PRIMARY_28 = "rgba(79,142,247,0.28)"
PRIMARY_80 = "rgba(79,142,247,0.80)"

# Annotation / reference lines
GRIDLINE = "#2A2D35"
REFLINE  = "#4A4D55"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply dark-mode legend and axis defaults to a Plotly figure."""
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE)
    return fig
