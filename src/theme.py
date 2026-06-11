"""Design tokens and Plotly chart styling — single source of truth for the visual system.

This module is pure Python (no Streamlit imports). Streamlit-dependent UI
components live in the root-level `ui.py`, which imports tokens from here so
charts and page chrome share one palette.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ── Surfaces ──────────────────────────────────────────────────────────────────
CANVAS         = "#0A0C10"   # page background
SURFACE        = "#11141B"   # panel background
SURFACE_RAISED = "#161A23"   # hover / nested surfaces
BORDER         = "#1E232E"   # hairline panel borders
BORDER_STRONG  = "#2A3140"   # hover / focus borders

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT       = "#E5E7EB"
TEXT_MUTED = "#8A92A6"
TEXT_FAINT = "#5C6470"

# ── Semantic series palette ───────────────────────────────────────────────────
PRIMARY   = "#4F8EF7"   # lines, bars, main series
BENCHMARK = "#E8A33D"   # comparison / benchmark series
POSITIVE  = "#3FB66B"   # gains, favorable outcomes
NEGATIVE  = "#E5564E"   # losses, adverse outcomes
NEUTRAL   = "#8A92A6"   # secondary / muted elements

# Opacity variants used in band fills (PRIMARY base)
PRIMARY_10 = "rgba(79,142,247,0.10)"
PRIMARY_18 = "rgba(79,142,247,0.18)"
PRIMARY_28 = "rgba(79,142,247,0.28)"
PRIMARY_80 = "rgba(79,142,247,0.80)"

NEGATIVE_18 = "rgba(229,86,78,0.18)"

# Annotation / reference lines
GRIDLINE = "#1C2029"
REFLINE  = "#4A4D55"

# ── Typography ────────────────────────────────────────────────────────────────
FONT_UI   = "Inter, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SF Mono', Menlo, monospace"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the terminal chart style: transparent surfaces, hairline grid,
    UI font for labels with monospace tick digits. Uses update_xaxes/update_yaxes
    so it works on multi-subplot figures."""
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family=FONT_UI, size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family=FONT_UI, size=12),
        hoverlabel=dict(
            bgcolor=SURFACE_RAISED,
            bordercolor=BORDER_STRONG,
            font=dict(color=TEXT, family=FONT_MONO, size=12),
        ),
    )
    # Style the title only when one exists — Plotly 6 renders the literal
    # string "undefined" if title_font is set on a figure with no title text.
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT_UI, size=13, color=TEXT_MUTED))
    fig.update_xaxes(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
        tickfont=dict(family=FONT_MONO, size=11, color=TEXT_MUTED),
        title_font=dict(family=FONT_UI, size=12, color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
        tickfont=dict(family=FONT_MONO, size=11, color=TEXT_MUTED),
        title_font=dict(family=FONT_UI, size=12, color=TEXT_MUTED),
    )
    fig.update_annotations(font=dict(family=FONT_UI, size=11, color=TEXT_MUTED))
    return fig


# Standard Plotly display config used by every page
CHART_CONFIG = {"responsive": True, "displayModeBar": False}
