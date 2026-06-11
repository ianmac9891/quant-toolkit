"""Relative Value Analysis — pair cointegration and spread diagnostics."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import data
from src import pairs as pr
from src.theme import (
    PRIMARY, BENCHMARK, NEGATIVE, NEUTRAL, REFLINE, TEXT,
    PRIMARY_18, NEGATIVE_18, CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Systematic Research", "Relative Value Analysis",
    "Pair diagnostics for relative-value research: OLS hedge ratio, "
    "Engle-Granger cointegration, spread half-life, and the current spread "
    "z-score. A statistical screen for mean-reversion candidates — past "
    "cointegration does not guarantee the relationship persists.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1.2])
    with c1:
        ticker_a = st.text_input("Instrument A (long leg)", value="XOM").upper().strip()
    with c2:
        ticker_b = st.text_input("Instrument B (hedge leg)", value="CVX").upper().strip()
    with c3:
        start_date, end_date = ui.date_range_input(
            "Estimation Window", today - timedelta(days=365 * 5), today,
        )
    with c4:
        entry_z = st.number_input(
            "Signal Threshold (z-score)", min_value=1.0, max_value=4.0,
            value=2.0, step=0.25,
            help="Reference bands drawn on the spread chart. A classical pairs "
                 "rule enters when |z| exceeds this level and exits near zero.",
        )

if not ticker_a or not ticker_b or ticker_a == ticker_b:
    ui.banner("info", "Enter two distinct instrument symbols to begin.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(ticker: str, start: date, end: date) -> pd.Series:
    df = data.get_prices(ticker, start, end)
    return df["adj_close"].dropna() if not df.empty else pd.Series(dtype=float)


with st.spinner(f"Retrieving {ticker_a} and {ticker_b}..."):
    pa = _fetch(ticker_a, start_date, end_date)
    pb = _fetch(ticker_b, start_date, end_date)

for t, s in ((ticker_a, pa), (ticker_b, pb)):
    if s.empty:
        ui.banner("error", f"No price data for <b>{t}</b>. Verify the symbol.")
        st.stop()

try:
    res = pr.analyze_pair(pa, pb, ticker_a, ticker_b)
except ValueError as e:
    ui.banner("error", str(e))
    st.stop()

# ── Assessment ────────────────────────────────────────────────────────────────

cointegrated = np.isfinite(res.coint_p) and res.coint_p < 0.05
hl_ok = np.isfinite(res.half_life_days)

if cointegrated:
    verdict = ui.tag(f"COINTEGRATED AT 5% (p = {res.coint_p:.4f})", "pos")
elif np.isfinite(res.coint_p):
    verdict = ui.tag(f"NOT COINTEGRATED (p = {res.coint_p:.4f})", "neu")
else:
    verdict = ui.tag("COINTEGRATION TEST UNAVAILABLE", "warn")

stretched = abs(res.current_z) >= entry_z

ui.kpi_row([
    {"label": "Hedge Ratio (log OLS)", "value": f"{res.hedge_ratio:.3f}"},
    {"label": "Engle-Granger p", "value": f"{res.coint_p:.4f}" if np.isfinite(res.coint_p) else "—"},
    {"label": "ADF p (spread)", "value": f"{res.adf_p:.4f}" if np.isfinite(res.adf_p) else "—"},
    {"label": "Half-Life", "value": f"{res.half_life_days:.0f} days" if hl_ok else "—"},
    {"label": "Current Z-Score", "value": f"{res.current_z:+.2f}",
     "delta_kind": "neg" if stretched else "neu"},
    {"label": "Return Correlation", "value": f"{res.return_corr:.2f}"},
])

with ui.panel("Statistical Assessment"):
    direction = (
        f"{res.ticker_a} rich vs {res.ticker_b}" if res.current_z > 0
        else f"{res.ticker_a} cheap vs {res.ticker_b}"
    )
    st.markdown(
        verdict + "&nbsp;&nbsp;" +
        (ui.tag(f"SPREAD AT {res.current_z:+.2f} SD — {direction.upper()}",
                "warn" if stretched else "neu")),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Sample: {res.n_obs:,} overlapping sessions. The Engle-Granger test "
        "accounts for the estimated hedge ratio and is the primary criterion; "
        "the ADF statistic on the spread is shown for reference. A finite "
        "half-life estimates how quickly spread deviations decay; the z-score "
        "standardizes the current spread against its full-sample distribution. "
        "Cointegration estimated in-sample frequently breaks down out of sample, "
        "particularly across regime changes."
    )

# ── Rebased prices ────────────────────────────────────────────────────────────

with ui.panel("Rebased Price History (both legs = 100 at window start)"):
    px_fig = go.Figure()
    px_fig.add_trace(go.Scatter(
        x=res.rebased_a.index, y=res.rebased_a.values, mode="lines",
        name=res.ticker_a, line=dict(color=PRIMARY, width=1.6),
    ))
    px_fig.add_trace(go.Scatter(
        x=res.rebased_b.index, y=res.rebased_b.values, mode="lines",
        name=res.ticker_b, line=dict(color=BENCHMARK, width=1.6),
    ))
    px_fig.update_layout(
        yaxis_title="Rebased price", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(px_fig)
    st.plotly_chart(px_fig, width="stretch", config=CHART_CONFIG)

# ── Spread ────────────────────────────────────────────────────────────────────

with ui.panel(f"Spread Z-Score — log({res.ticker_a}) − {res.hedge_ratio:.3f} · log({res.ticker_b})"):
    z = res.zscore
    sp_fig = go.Figure()

    sp_fig.add_hrect(y0=entry_z, y1=max(float(z.max()), entry_z) + 0.5,
                     fillcolor=NEGATIVE_18, line_width=0)
    sp_fig.add_hrect(y0=min(float(z.min()), -entry_z) - 0.5, y1=-entry_z,
                     fillcolor=PRIMARY_18, line_width=0)

    sp_fig.add_trace(go.Scatter(
        x=z.index, y=z.values, mode="lines",
        name="Spread z-score", line=dict(color=TEXT, width=1.3),
    ))
    sp_fig.add_hline(y=0, line_color=REFLINE, line_width=1)
    for lvl in (entry_z, -entry_z):
        sp_fig.add_hline(y=lvl, line_dash="dash", line_color=NEUTRAL, line_width=1)
    sp_fig.add_trace(go.Scatter(
        x=[z.index[-1]], y=[res.current_z], mode="markers",
        marker=dict(size=10, color=BENCHMARK, symbol="diamond"),
        name="Current",
    ))
    sp_fig.update_layout(
        yaxis_title="Z-score (full-sample)", height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", showlegend=False,
    )
    apply_chart_theme(sp_fig)
    st.plotly_chart(sp_fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        f"Shaded regions mark |z| beyond the {entry_z:g}-standard-deviation "
        "threshold. Positive z: instrument A is rich relative to the hedge "
        "(classical setup shorts A, buys B at the hedge ratio); negative z is "
        "the reverse. Z-scores use the full-sample mean and standard deviation, "
        "so early values are computed with information not available at the time."
    )

# ── Methodology ───────────────────────────────────────────────────────────────

with st.expander("Methodology and Limitations"):
    st.markdown("""
**Hedge ratio.** OLS of log prices: log A = c + h·log B. The residual is the
spread; h is the position ratio (per $1 of A short, h·$1 of B long, in log
terms).

**Cointegration.** Engle-Granger two-step test. Its critical values account for
the estimated hedge ratio, making it the primary criterion. The ADF test on the
spread uses standard critical values and is reported for reference only.

**Half-life.** AR(1) regression of spread changes on the lagged spread level;
half-life = −ln 2 / ln(1 + b). Undefined (shown as —) when the coefficient is
non-negative, i.e. the spread shows no estimated mean reversion.

**Limitations.**
- In-sample hedge ratios drift; rolling re-estimation is standard in production.
- Cointegration found by scanning many pairs is subject to severe selection
  bias — at a 5% threshold, one in twenty random pairs qualifies by chance.
- The z-score uses full-sample moments and therefore overstates how stretched
  the spread would have looked in real time.
- No transaction costs, borrow costs, or execution constraints are modeled.
""")

ui.footer_disclaimer()
