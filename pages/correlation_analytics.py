"""Correlation Analytics — cross-asset correlation structure and diversification."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import correlation as cr
from src.theme import PRIMARY, BENCHMARK, NEUTRAL, REFLINE, CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Portfolio & Risk", "Correlation Analytics",
    "Cross-asset correlation structure: the full-sample matrix, the rolling "
    "average pairwise correlation (the diversification pulse), and "
    "concentration gauges. Correlations rise in stress episodes — "
    "diversification tends to fail exactly when it is needed.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2 = st.columns([1, 1.8])
    with c1:
        tickers = ui.ticker_list_input(
            "Universe", "SPY\nQQQ\nIWM\nTLT\nGLD\nHYG\nUUP", height=150,
            help="Two to roughly twenty instruments. Mixed asset classes show "
                 "the diversification structure best.",
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            start_date, end_date = ui.date_range_input(
                "Observation Window", today - timedelta(days=365 * 5), today,
            )
        with cc2:
            roll_window = st.select_slider(
                "Rolling Window (sessions)", options=[21, 42, 63, 126, 252], value=63,
                help="63 sessions is approximately one quarter.",
            )

if len(tickers) < 2:
    ui.banner("warn", "Specify at least two instruments.")
    st.stop()

if len(tickers) > 25:
    ui.banner("warn", f"{len(tickers)} instruments requested — limiting to the "
                      "first 25 for readability.")
    tickers = tickers[:25]

# ── Data ──────────────────────────────────────────────────────────────────────

with st.spinner("Retrieving price histories..."):
    frames = ui.fetch_universe(tuple(sorted(tickers)), start_date, end_date)

cols = {
    t: df["adj_close"].rename(t)
    for t, df in frames.items()
    if not df.empty and "adj_close" in df.columns
}
price_df = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()

failed = sorted(set(tickers) - set(price_df.columns))
if failed:
    ui.banner("warn", f"No data for: <span class='mono'>{', '.join(failed)}</span> — excluded.")
if price_df.empty:
    ui.data_unavailable()
    st.stop()

price_df = price_df.dropna()
if price_df.shape[1] < 2 or len(price_df) < 60:
    ui.banner("error", "At least two instruments with 60+ overlapping sessions are required.")
    st.stop()

ui.data_asof_caption(price_df.index.max())

returns_df = price_df.pct_change().dropna()
active = list(returns_df.columns)

# ── Headline gauges ───────────────────────────────────────────────────────────

corr = cr.correlation_matrix(returns_df)
avg_corr = cr.mean_offdiag_correlation(corr)
pc1 = cr.pc1_variance_share(corr)
div_ratio = cr.diversification_ratio(returns_df)
high_pairs, low_pairs = cr.extreme_pairs(corr, k=1)

ui.kpi_row([
    {"label": "Instruments", "value": f"{len(active)}"},
    {"label": "Avg Pairwise Correlation", "value": f"{avg_corr:.2f}"},
    {"label": "PC1 Variance Share", "value": f"{pc1:.0%}"},
    {"label": "Diversification Ratio (EW)", "value": f"{div_ratio:.2f}"},
    {"label": "Highest Pair",
     "value": f"{high_pairs[0][2]:.2f}" if high_pairs else "—",
     "delta": f"{high_pairs[0][0]} / {high_pairs[0][1]}" if high_pairs else None,
     "delta_kind": "neu"},
    {"label": "Lowest Pair",
     "value": f"{low_pairs[0][2]:.2f}" if low_pairs else "—",
     "delta": f"{low_pairs[0][0]} / {low_pairs[0][1]}" if low_pairs else None,
     "delta_kind": "neu"},
])

# ── Correlation matrix ────────────────────────────────────────────────────────

with ui.panel(f"Return Correlation Matrix — {len(returns_df):,} sessions"):
    hm = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
    ))
    hm.update_layout(
        height=max(360, 32 * len(active) + 80),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    apply_chart_theme(hm)
    st.plotly_chart(hm, width="stretch", config=CHART_CONFIG)

# ── Rolling average correlation ───────────────────────────────────────────────

with ui.panel(f"Rolling Average Pairwise Correlation ({roll_window} sessions)"):
    with st.spinner("Computing rolling correlation structure..."):
        mean_roll = cr.rolling_mean_correlation(returns_df, window=roll_window)

    if mean_roll.empty:
        ui.banner("info", "Insufficient history for the selected rolling window.")
    else:
        rc_fig = go.Figure()
        rc_fig.add_trace(go.Scatter(
            x=mean_roll.index, y=mean_roll.values, mode="lines",
            line=dict(color=PRIMARY, width=1.6), name="Average pairwise correlation",
        ))
        rc_fig.add_hline(
            y=avg_corr, line_dash="dash", line_color=NEUTRAL, line_width=1.2,
            annotation_text=f"Full sample {avg_corr:.2f}",
            annotation_position="top left", annotation_font_size=11,
        )
        rc_fig.add_hline(y=0, line_color=REFLINE, line_width=1)
        rc_fig.update_layout(
            yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", showlegend=False,
        )
        apply_chart_theme(rc_fig)
        st.plotly_chart(rc_fig, width="stretch", config=CHART_CONFIG)
        st.caption(
            "Spikes mark periods when the universe moved as one — typically "
            "drawdowns — and the diversification benefit measured over the full "
            "sample temporarily disappeared."
        )

# ── Pair drill-down ───────────────────────────────────────────────────────────

with ui.panel("Pair Drill-Down"):
    pc1_col, pc2_col = st.columns(2)
    with pc1_col:
        pair_a = st.selectbox("Instrument A", active, index=0)
    with pc2_col:
        pair_b = st.selectbox("Instrument B", active, index=min(1, len(active) - 1))

    if pair_a == pair_b:
        ui.banner("info", "Select two distinct instruments.")
    else:
        pair_roll = cr.rolling_pair_correlation(returns_df, pair_a, pair_b, window=roll_window)
        full_pair = float(corr.loc[pair_a, pair_b])

        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(
            x=pair_roll.index, y=pair_roll.values, mode="lines",
            line=dict(color=BENCHMARK, width=1.6),
            name=f"{pair_a} / {pair_b}",
        ))
        pr_fig.add_hline(
            y=full_pair, line_dash="dash", line_color=NEUTRAL, line_width=1.2,
            annotation_text=f"Full sample {full_pair:.2f}",
            annotation_position="top left", annotation_font_size=11,
        )
        pr_fig.add_hline(y=0, line_color=REFLINE, line_width=1)
        pr_fig.update_layout(
            yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", showlegend=False,
        )
        apply_chart_theme(pr_fig)
        st.plotly_chart(pr_fig, width="stretch", config=CHART_CONFIG)

# ── Notes ─────────────────────────────────────────────────────────────────────

with st.expander("Reading These Gauges"):
    st.markdown("""
**Average pairwise correlation** — the mean of all off-diagonal entries. Lower
is better for diversification; equity-only universes typically sit between 0.4
and 0.7, while mixed asset classes can approach zero.

**PC1 variance share** — the fraction of total variance explained by the first
principal component of the correlation matrix. Values near 1/N indicate evenly
spread risk; values approaching 100% indicate one common factor drives
everything (a single-factor market).

**Diversification ratio** — weighted average asset volatility divided by
portfolio volatility (equal weights here). 1.0 means combining the assets
reduced nothing; the gap above 1.0 is the volatility eliminated by imperfect
correlation.

**Caveats.** Pearson correlation captures linear co-movement only and is highly
regime-dependent: tail correlations are systematically higher than full-sample
estimates. Rolling windows lag regime changes by design.
""")

ui.footer_disclaimer()
