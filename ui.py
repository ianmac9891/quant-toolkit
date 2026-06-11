"""Quant Research Terminal — design system and shared UI components.

All Streamlit-dependent chrome lives here: the injected CSS design system,
page headers, panels, KPI strips, status banners, and validated input helpers.
Design tokens (colors, fonts) are imported from src/theme.py so charts and
chrome share one palette. Pages should never call st.metric / st.info /
st.warning / st.error directly — use kpi_row() and banner() instead.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from datetime import date

import streamlit as st
import streamlit.components.v1 as components

from src import theme as tk

APP_NAME = "Quant Research Terminal"
APP_WORDMARK = "QRT"

# ──────────────────────────────────────────────────────────────────────────────
# Design system CSS — injected once per run from app.py
# ──────────────────────────────────────────────────────────────────────────────

_DESIGN_SYSTEM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --canvas: {tk.CANVAS};
    --surface: {tk.SURFACE};
    --surface-raised: {tk.SURFACE_RAISED};
    --border: {tk.BORDER};
    --border-strong: {tk.BORDER_STRONG};
    --text: {tk.TEXT};
    --text-muted: {tk.TEXT_MUTED};
    --text-faint: {tk.TEXT_FAINT};
    --accent: {tk.PRIMARY};
    --benchmark: {tk.BENCHMARK};
    --positive: {tk.POSITIVE};
    --negative: {tk.NEGATIVE};
    --font-ui: {tk.FONT_UI};
    --font-mono: {tk.FONT_MONO};
}}

/* ── Base canvas and typography ── */
.stApp {{ background: var(--canvas); }}
html, body, .stApp, .stMarkdown, p, li {{ font-family: var(--font-ui); }}
.stMarkdown p, .stMarkdown li {{ font-size: 13.5px; line-height: 1.55; color: var(--text); }}

.block-container {{
    max-width: 1180px;
    padding-top: 1.1rem;
    padding-bottom: max(3rem, env(safe-area-inset-bottom));
}}

/* Hide chrome we replace: header decoration, toolbar, sidebar toggle */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

h1, h2, h3 {{ font-family: var(--font-ui); color: var(--text); }}
h1 {{ font-size: 24px !important; font-weight: 650 !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 17px !important; font-weight: 600 !important; }}
h3 {{ font-size: 15px !important; font-weight: 600 !important; }}

[data-testid="stCaptionContainer"] p {{
    color: var(--text-faint) !important;
    font-size: 12px !important;
    line-height: 1.5;
}}

hr {{ border-color: var(--border) !important; margin: 1.2rem 0 !important; }}

/* ── Panels: bordered containers and forms ── */
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 6px;
    padding: 1.05rem 1.2rem;
}}
[data-testid="stForm"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.05rem 1.2rem;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input {{
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    background: var(--surface-raised) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: var(--surface-raised) !important;
    border-color: var(--border) !important;
    font-size: 13px;
}}
.stTextInput label p, .stNumberInput label p, .stTextArea label p,
.stDateInput label p, .stSelectbox label p, .stSlider label p,
.stRadio label p, .stCheckbox label p, .stMultiSelect label p {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}}
.stRadio [role="radiogroup"] p, .stCheckbox [data-testid="stWidgetLabel"] ~ * p {{
    text-transform: none;
    letter-spacing: 0;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: var(--text) !important;
}}

/* ── Buttons ── */
.stButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button {{
    font-family: var(--font-ui) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 4px !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--surface-raised) !important;
    color: var(--text) !important;
    min-height: 38px;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}}
[data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #0A0C10 !important;
}}
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.12);
}}

/* ── Tables ── */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface);
}}
[data-testid="stExpander"] summary p {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}}

/* ── Custom components ── */
.qrt-topbar {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.65rem;
    margin-bottom: 1.4rem;
}}
.qrt-topbar a {{
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--text) !important;
    text-decoration: none !important;
}}
.qrt-topbar a span {{ color: var(--accent); }}
.qrt-topbar a:hover {{ color: var(--accent) !important; }}
.qrt-topbar .qrt-section {{
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-faint);
}}

.qrt-pagetitle {{ margin: 0 0 0.2rem 0; font-size: 24px; font-weight: 650; letter-spacing: -0.01em; color: var(--text); }}
.qrt-pagedesc  {{ margin: 0 0 1.3rem 0; font-size: 13px; color: var(--text-muted); max-width: 760px; line-height: 1.5; }}

.qrt-kicker {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 0.55rem 0;
}}

.qrt-kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
    gap: 10px;
    margin: 0.2rem 0 1rem 0;
}}
.qrt-kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
}}
.qrt-kpi .lbl {{
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.qrt-kpi .val {{
    font-family: var(--font-mono);
    font-size: 21px;
    font-weight: 500;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
}}
.qrt-kpi .delta {{
    font-family: var(--font-mono);
    font-size: 11.5px;
    margin-top: 0.25rem;
    font-variant-numeric: tabular-nums;
}}
.qrt-kpi .delta.pos {{ color: var(--positive); }}
.qrt-kpi .delta.neg {{ color: var(--negative); }}
.qrt-kpi .delta.neu {{ color: var(--text-faint); }}

.qrt-banner {{
    border: 1px solid var(--border);
    border-left-width: 3px;
    border-radius: 4px;
    background: var(--surface);
    padding: 0.7rem 1rem;
    font-size: 13px;
    line-height: 1.55;
    color: var(--text);
    margin: 0.4rem 0 1rem 0;
}}
.qrt-banner.info    {{ border-left-color: var(--accent); }}
.qrt-banner.warn    {{ border-left-color: var(--benchmark); }}
.qrt-banner.error   {{ border-left-color: var(--negative); }}
.qrt-banner.success {{ border-left-color: var(--positive); }}
.qrt-banner b, .qrt-banner strong {{ font-weight: 600; }}
.qrt-banner .mono {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}

.qrt-card {{
    display: block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.05rem 1.15rem;
    text-decoration: none !important;
    height: 100%;
    transition: border-color 120ms ease, background 120ms ease;
}}
.qrt-card:hover {{
    border-color: var(--accent);
    background: var(--surface-raised);
}}
.qrt-card .t {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
}}
.qrt-card .d {{
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.5;
    margin-bottom: 0.6rem;
}}
.qrt-card .caps {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-faint);
    line-height: 1.7;
}}

.qrt-footer {{
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    padding-top: 0.9rem;
    font-size: 11px;
    line-height: 1.6;
    color: var(--text-faint);
}}

.qrt-tag {{
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 2px 9px;
    border-radius: 3px;
    border: 1px solid var(--border-strong);
}}
.qrt-tag.pos {{ color: var(--positive); border-color: var(--positive); }}
.qrt-tag.neg {{ color: var(--negative); border-color: var(--negative); }}
.qrt-tag.neu {{ color: var(--text-muted); }}
.qrt-tag.accent {{ color: var(--accent); border-color: var(--accent); }}
.qrt-tag.warn {{ color: var(--benchmark); border-color: var(--benchmark); }}

/* ── Mobile (≤ 767px) ── */
@media (max-width: 767px) {{
    .block-container {{
        padding-top: 0.75rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }}
    [data-testid="column"] {{
        width: 100% !important;
        flex: 0 0 100% !important;
        min-width: 100% !important;
    }}
    [data-testid="stDataFrame"], .stPlotlyChart {{
        overflow-x: auto !important;
        max-width: 100% !important;
    }}
    input, select, textarea {{ font-size: 16px !important; }}
    .stButton > button {{ min-height: 44px !important; }}
    .qrt-kpi-row {{ grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }}
}}
@media (hover: none) {{
    .modebar-container {{ display: none !important; }}
}}
</style>
"""


def inject_design_system() -> None:
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)


def inject_pwa() -> None:
    """Head injection for iOS/Android home-screen install. Runs inside a
    same-origin component iframe so the script actually executes (scripts in
    st.markdown HTML are inert), reaching the parent document via window.parent."""
    components.html(
        f"""
        <script>
        (function () {{
            var doc = window.parent.document;
            var head = doc.querySelector('head');
            if (!head || head.querySelector('meta[name="apple-mobile-web-app-capable"]')) return;
            var APP_NAME = '{APP_NAME}';
            function meta(n, c) {{ var m = doc.createElement('meta'); m.name = n; m.content = c; head.appendChild(m); }}
            function link(rel, href, extra) {{
                var l = doc.createElement('link'); l.rel = rel; l.href = href;
                if (extra) Object.assign(l, extra); head.appendChild(l);
            }}
            meta('apple-mobile-web-app-capable', 'yes');
            meta('apple-mobile-web-app-status-bar-style', 'black');
            meta('apple-mobile-web-app-title', APP_NAME);
            meta('mobile-web-app-capable', 'yes');
            meta('theme-color', '{tk.CANVAS}');
            link('apple-touch-icon', '/app/static/icon-180.png', {{sizes: '180x180'}});
            link('manifest', '/app/static/manifest.json');
            var vp = head.querySelector('meta[name="viewport"]');
            if (vp) vp.content = 'width=device-width, initial-scale=1, shrink-to-fit=no, viewport-fit=cover';
            doc.title = APP_NAME;
            var titleEl = doc.querySelector('title');
            if (titleEl) {{
                new MutationObserver(function () {{
                    if (doc.title !== APP_NAME) doc.title = APP_NAME;
                }}).observe(titleEl, {{ childList: true, characterData: true, subtree: true }});
            }}
        }})();
        </script>
        """,
        height=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout components
# ──────────────────────────────────────────────────────────────────────────────

def page_header(section: str, title: str, description: str = "") -> None:
    """Top bar (wordmark routes home) + page title + mandate line."""
    st.markdown(
        f"""
        <div class="qrt-topbar">
            <a href="/" target="_self"><span>{APP_WORDMARK}</span>&nbsp;&nbsp;{APP_NAME.upper()}</a>
            <span class="qrt-section">{html.escape(section)}</span>
        </div>
        <div class="qrt-pagetitle">{html.escape(title)}</div>
        """
        + (f'<div class="qrt-pagedesc">{html.escape(description)}</div>' if description else ""),
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Uppercase section label used between panels."""
    st.markdown(f'<p class="qrt-kicker" style="margin-top:1.1rem">{html.escape(title)}</p>',
                unsafe_allow_html=True)


@contextmanager
def panel(title: str | None = None):
    """Bordered surface container with an optional kicker label."""
    with st.container(border=True):
        if title:
            st.markdown(f'<p class="qrt-kicker">{html.escape(title)}</p>', unsafe_allow_html=True)
        yield


def kpi_row(items: list[dict]) -> None:
    """Render a strip of KPI cells.

    Each item: {label, value, delta (optional, pre-formatted str),
                delta_kind (optional: 'pos'|'neg'|'neu' — explicit coloring)}
    If delta_kind is omitted it is inferred from the delta's leading sign.
    """
    cells = []
    for it in items:
        delta_html = ""
        delta = it.get("delta")
        if delta is not None:
            kind = it.get("delta_kind")
            if kind is None:
                kind = "pos" if str(delta).startswith("+") else ("neg" if str(delta).startswith("-") else "neu")
            delta_html = f'<div class="delta {kind}">{html.escape(str(delta))}</div>'
        cells.append(
            f'<div class="qrt-kpi"><div class="lbl" title="{html.escape(str(it["label"]))}">'
            f'{html.escape(str(it["label"]))}</div>'
            f'<div class="val">{html.escape(str(it["value"]))}</div>{delta_html}</div>'
        )
    st.markdown(f'<div class="qrt-kpi-row">{"".join(cells)}</div>', unsafe_allow_html=True)


def banner(kind: str, body: str) -> None:
    """Status banner. kind: info | warn | error | success. Body may contain
    <b>/<span class='mono'> markup — caller is responsible for escaping data."""
    st.markdown(f'<div class="qrt-banner {kind}">{body}</div>', unsafe_allow_html=True)


def tag(text: str, kind: str = "neu") -> str:
    """Inline status tag HTML (returned, not rendered). kind: pos|neg|neu|accent|warn."""
    return f'<span class="qrt-tag {kind}">{html.escape(text)}</span>'


def nav_card(url_path: str, title: str, description: str, capabilities: list[str]) -> str:
    """HTML card linking to a registered page (hidden navigation keeps URLs routable)."""
    caps = "<br>".join(html.escape(c) for c in capabilities)
    return (
        f'<a class="qrt-card" href="/{url_path}" target="_self">'
        f'<div class="t">{html.escape(title)}</div>'
        f'<div class="d">{html.escape(description)}</div>'
        f'<div class="caps">{caps}</div></a>'
    )


def footer_disclaimer() -> None:
    st.markdown(
        """
        <div class="qrt-footer">
        FOR RESEARCH AND EDUCATIONAL USE ONLY. This application does not constitute investment
        advice, a solicitation, or an offer to buy or sell any security. The author is not a
        licensed financial advisor, registered investment advisor, or broker-dealer in any
        jurisdiction. All analytics are derived from third-party historical data that may contain
        errors, omissions, or delays; past performance does not guarantee future results. All
        liability for decisions made in reliance on this application is disclaimed to the fullest
        extent permitted by law. Consult a qualified, licensed financial professional before
        making investment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validated input helpers
# ──────────────────────────────────────────────────────────────────────────────

def date_range_input(
    label: str,
    default_start: date,
    default_end: date,
    min_value: date = date(1990, 1, 1),
    max_value: date | None = None,
    key: str | None = None,
    help: str | None = None,
) -> tuple[date, date]:
    """st.date_input range wrapper that handles the mid-selection state (one
    date picked, second pending) instead of crashing on tuple unpack."""
    result = st.date_input(
        label,
        value=(default_start, default_end),
        min_value=min_value,
        max_value=max_value or date.today(),
        key=key,
        help=help,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        banner("info", "Select both a start and an end date to continue.")
        st.stop()
    return result  # type: ignore[return-value]


def rf_rate_input(key: str | None = None, default_pct: float = 4.5) -> float:
    """Standard risk-free rate input. Entered in percent, returned as a decimal.
    One convention across every tool."""
    pct = st.number_input(
        "Risk-Free Rate (% per annum)",
        min_value=0.0, max_value=20.0, value=default_pct, step=0.25,
        key=key,
        help="Annualized risk-free rate used for excess-return calculations "
             "(Sharpe, Sortino, capital market line). A standard proxy is the "
             "3-month Treasury bill yield.",
    )
    return pct / 100.0


def ticker_list_input(label: str, default: str, height: int = 120, key: str | None = None,
                      help: str | None = None) -> list[str]:
    """Multi-ticker text area → sorted, deduplicated, uppercased list."""
    import re
    raw = st.text_area(label, default, height=height, key=key,
                       help=help or "One symbol per line, or comma-separated.")
    return sorted(set(t for t in re.split(r"[\s,]+", raw.strip().upper()) if t))
