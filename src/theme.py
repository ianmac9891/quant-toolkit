"""Design tokens and Plotly chart styling — single source of truth for the visual system.

This module is pure Python (no Streamlit imports). Streamlit-dependent UI
components live in the root-level `ui.py`, which imports tokens from here so
charts and page chrome share one palette.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ── Surfaces — warm graphite, no blue cast ────────────────────────────────────
CANVAS         = "#131211"   # page background
SURFACE        = "#1B1A18"   # panel background
SURFACE_RAISED = "#232120"   # hover / nested surfaces
BORDER         = "#2C2925"   # hairline panel borders
BORDER_STRONG  = "#3E3931"   # hover / focus borders

# ── Text — warm off-white ─────────────────────────────────────────────────────
TEXT       = "#EDEAE3"
TEXT_MUTED = "#A39C8E"
TEXT_FAINT = "#6E685C"

# ── Semantic series palette — amber primary, steel benchmark ──────────────────
PRIMARY   = "#E2A33D"   # lines, bars, main series, UI accent
BENCHMARK = "#8FA8C8"   # comparison / benchmark series (muted steel)
POSITIVE  = "#5FA97C"   # gains, favorable outcomes
NEGATIVE  = "#D2625A"   # losses, adverse outcomes
NEUTRAL   = "#8F897D"   # secondary / muted elements

# Opacity variants used in band fills (PRIMARY base)
PRIMARY_10 = "rgba(226,163,61,0.10)"
PRIMARY_18 = "rgba(226,163,61,0.18)"
PRIMARY_28 = "rgba(226,163,61,0.28)"
PRIMARY_80 = "rgba(226,163,61,0.85)"

NEGATIVE_18 = "rgba(210,98,90,0.16)"

# Annotation / reference lines
GRIDLINE = "#262420"
REFLINE  = "#4E483E"

# Diverging heatmap endpoints (deep loss red → surface → deep gain green)
HEAT_NEG = "#54302B"
HEAT_POS = "#2C4A38"

# ── Typography ────────────────────────────────────────────────────────────────
FONT_DISPLAY = "'Newsreader', 'Iowan Old Style', Georgia, serif"   # titles, masthead
FONT_UI      = "Inter, -apple-system, 'Segoe UI', sans-serif"      # body, labels
FONT_MONO    = "'IBM Plex Mono', 'SF Mono', Menlo, monospace"      # numerals only


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the terminal chart style: transparent surfaces, hairline grid,
    UI font for labels with monospace tick digits. Uses update_xaxes/update_yaxes
    so it works on multi-subplot figures."""
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family=FONT_UI, size=13),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family=FONT_UI, size=13),
        hoverlabel=dict(
            bgcolor=SURFACE_RAISED,
            bordercolor=BORDER_STRONG,
            font=dict(color=TEXT, family=FONT_MONO, size=12),
        ),
    )
    # Style the title only when one exists — Plotly 6 renders the literal
    # string "undefined" if title_font is set on a figure with no title text.
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT_UI, size=14, color=TEXT_MUTED))
    fig.update_xaxes(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
        tickfont=dict(family=FONT_MONO, size=12, color=TEXT_MUTED),
        title_font=dict(family=FONT_UI, size=13, color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
        tickfont=dict(family=FONT_MONO, size=12, color=TEXT_MUTED),
        title_font=dict(family=FONT_UI, size=13, color=TEXT_MUTED),
    )
    fig.update_annotations(font=dict(family=FONT_UI, size=12, color=TEXT_MUTED))
    return fig


# Standard Plotly display config used by every page
CHART_CONFIG = {"responsive": True, "displayModeBar": False}
