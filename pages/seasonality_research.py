"""Seasonality Research — calendar-effect hypothesis testing with FDR control."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import anomalies as an
from src.theme import PRIMARY, BENCHMARK, NEUTRAL, CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Systematic Research", "Seasonality Research",
    "Tests whether calendar patterns predict positive-close days, with "
    "Benjamini-Hochberg false-discovery-rate control, out-of-sample "
    "replication requirements, and net-of-cost economics. Designed as a "
    "skepticism aid, not a signal generator.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2, c3 = st.columns([1, 1.3, 1.3])
    with c1:
        ticker = st.text_input("Instrument", value=ui.get_default_ticker("SPY")).upper().strip()
    with c2:
        start_date = st.date_input(
            "Sample Start",
            value=date(today.year - 15, today.month, today.day),
            min_value=date(2000, 1, 1),
            max_value=today - timedelta(days=365),
            help="Fifteen years provides roughly 3,750 observations — the minimum "
                 "for reasonable statistical power on the rarer calendar buckets.",
        )
    with c3:
        cost_bps = st.slider(
            "Transaction Cost (bps, one-way)", min_value=1, max_value=20, value=5, step=1,
            help="Used for the net-of-cost return assessment. Does not affect the "
                 "primary hypothesis tests.",
        )
end_date = today

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _run_lab(prices: pd.Series, cost_bps: float) -> list[an.AnomalyCategory]:
    return an.run_anomaly_lab(prices, cost_bps=cost_bps)


# ── Fetch and run ─────────────────────────────────────────────────────────────

if not ticker:
    ui.banner("info", "Enter an instrument symbol to begin.")
    st.stop()

with st.spinner(f"Retrieving {ticker} price history..."):
    result = ui.fetch_prices(ticker, start_date, end_date)

if not result.ok or "adj_close" not in result.df.columns:
    ui.data_unavailable(f"{ticker}: {result.error or 'no usable columns'}")
    st.stop()
price_df = result.df
ui.remember_ticker(ticker)
ui.data_asof_caption(result.asof, result.source)

prices = price_df["adj_close"]
n_days = len(prices) - 1

if n_days < 500:
    ui.banner("warn", f"Only {n_days} sessions available — extend the sample for "
                      "reliable estimates.")

with st.spinner("Running hypothesis tests..."):
    categories = _run_lab(prices, float(cost_bps))

all_hyps = [h for cat in categories for h in cat.hypotheses]

n_total    = len(all_hyps)
n_raw_sig  = sum(1 for h in all_hyps if not np.isnan(h.p_raw) and h.p_raw < 0.05)
n_fdr_sig  = sum(1 for h in all_hyps if not np.isnan(h.p_fdr) and h.p_fdr < 0.05)
n_real     = sum(1 for h in all_hyps if h.verdict == "Real pattern")
n_tradable = sum(1 for h in all_hyps if h.tradable)

baseline_green = float(prices.pct_change().dropna().gt(0).mean())

# ── Summary ───────────────────────────────────────────────────────────────────

st.caption(f"{ticker} · {start_date} to {end_date} · {n_days:,} sessions")

ui.kpi_row([
    {"label": "Hypotheses Tested", "value": f"{n_total}"},
    {"label": "Raw p < 0.05", "value": f"{n_raw_sig}"},
    {"label": "Survive FDR Control", "value": f"{n_fdr_sig}"},
    {"label": "Validated Out-of-Sample", "value": f"{n_real}"},
    {"label": "Positive Net of Costs", "value": f"{n_tradable}"},
    {"label": "Baseline Positive-Close Rate", "value": f"{baseline_green*100:.1f}%"},
])

with ui.panel("The Multiple-Testing Problem"):
    st.markdown(f"""
Testing enough calendar hypotheses guarantees that some appear statistically
significant by chance: with {n_total} tests at a 5% threshold, one false positive
is expected under the null of no effect. The Benjamini-Hochberg correction
adjusts p-values for the number of simultaneous tests, applied here across all
{n_total} hypotheses in a single pass.

The stated count still understates the true data-mining burden. Window
definitions — three sessions around the new moon, the last session of a month
plus the first three of the next — were chosen after reviewing the academic
literature, and each such design decision introduces hidden degrees of freedom
that formal corrections cannot reach.
""")

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _category_tag(cat: an.AnomalyCategory) -> str:
    n_real_cat = sum(1 for h in cat.hypotheses if h.verdict == "Real pattern")
    n_tot_cat  = len(cat.hypotheses)
    if n_real_cat == 0:
        return ui.tag("NOT VALIDATED — CONSISTENT WITH NOISE", "neu")
    return ui.tag(f"{n_real_cat} OF {n_tot_cat} VALIDATED", "accent")


def _green_rate_chart(bucket_stats, signal_labels, baseline) -> go.Figure:
    labels      = [b.label for b in bucket_stats]
    green_rates = [b.green_rate * 100 for b in bucket_stats]
    se_vals     = [b.se_green * 100 for b in bucket_stats]
    colors      = [PRIMARY if lbl in signal_labels else NEUTRAL for lbl in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=green_rates,
        error_y=dict(type="data", array=se_vals, visible=True,
                     color=NEUTRAL, thickness=1.5, width=4),
        marker_color=colors,
        text=[f"{g:.1f}%" for g in green_rates],
        textposition="outside",
        hovertemplate="%{x}<br>Positive-close rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=baseline * 100, line_dash="dash", line_color=BENCHMARK, line_width=1.5,
        annotation_text=f"Baseline {baseline*100:.1f}%",
        annotation_position="top right", annotation_font_size=11,
    )
    lo = max(0,   min(green_rates) - 4)
    hi = min(100, max(green_rates) + 6)
    fig.update_layout(
        yaxis_title="Positive-close rate (%)",
        yaxis=dict(range=[lo, hi]),
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


def _hypothesis_df(hyps: list[an.HypothesisResult]) -> pd.DataFrame:
    rows = []
    for h in hyps:
        rows.append({
            "Signal":             h.signal_label,
            "N (signal)":         h.n_signal,
            "Positive % signal":  h.green_rate_signal,
            "Positive % other":   h.green_rate_other,
            "Gap (pp)":           h.green_rate_gap * 100 if not np.isnan(h.green_rate_gap) else float("nan"),
            "Z":                  h.z_stat,
            "Raw p":              h.p_raw,
            "FDR p":              h.p_fdr,
            "IS gap (pp)":        h.is_green_rate_gap  * 100 if not np.isnan(h.is_green_rate_gap)  else float("nan"),
            "OOS gap (pp)":       h.oos_green_rate_gap * 100 if not np.isnan(h.oos_green_rate_gap) else float("nan"),
            "Mean return (bps)":  h.mean_return_bps,
            "Net of costs (bps)": h.post_cost_mean_return_bps,
            "Assessment":         "Validated" if h.verdict == "Real pattern" else "Not validated",
            "Net Positive":       "Yes" if h.tradable else "No",
        })
    return pd.DataFrame(rows)


_TABLE_CFG = {
    "Signal":             st.column_config.TextColumn("Signal"),
    "N (signal)":         st.column_config.NumberColumn("N", format="%d"),
    "Positive % signal":  st.column_config.NumberColumn("Positive % Signal", format="%.1f%%"),
    "Positive % other":   st.column_config.NumberColumn("Positive % Other", format="%.1f%%"),
    "Gap (pp)":           st.column_config.NumberColumn("Gap (pp)", format="%+.2f",
                              help="Positive-close rate on signal days minus all other "
                                   "days, in percentage points"),
    "Z":                  st.column_config.NumberColumn("Z", format="%.2f"),
    "Raw p":              st.column_config.NumberColumn("Raw p", format="%.3f"),
    "FDR p":              st.column_config.NumberColumn("FDR p", format="%.3f",
                              help="Benjamini-Hochberg corrected p-value across all tests"),
    "IS gap (pp)":        st.column_config.NumberColumn("IS Gap (pp)", format="%+.2f",
                              help="Gap in the first half of the sample"),
    "OOS gap (pp)":       st.column_config.NumberColumn("OOS Gap (pp)", format="%+.2f",
                              help="Gap in the second half of the sample"),
    "Mean return (bps)":  st.column_config.NumberColumn("Mean Return (bps)", format="%.1f",
                              help="Mean daily return on signal days, basis points"),
    "Net of costs (bps)": st.column_config.NumberColumn("Net of Costs (bps)", format="%.1f",
                              help="Mean return after round-trip transaction costs"),
    "Assessment":         st.column_config.TextColumn("Assessment"),
    "Net Positive":       st.column_config.TextColumn("Net Positive"),
}

# ── Categories ────────────────────────────────────────────────────────────────

for cat in categories:
    with ui.panel(cat.name.upper()):
        st.markdown(_category_tag(cat) + f"&nbsp;&nbsp;{cat.description}",
                    unsafe_allow_html=True)

        signal_labels = {h.signal_label for h in cat.hypotheses}
        st.plotly_chart(
            _green_rate_chart(cat.bucket_stats, signal_labels, baseline_green),
            width="stretch", config=CHART_CONFIG,
        )

        with st.expander("Detailed Statistics"):
            df = _hypothesis_df(cat.hypotheses)
            if df.empty:
                ui.banner("info", "No testable signal buckets (insufficient observations).")
            else:
                st.dataframe(df, column_config=_TABLE_CFG,
                             width="stretch", hide_index=True)
                ui.download_row(df, f"seasonality_{cat.name.lower().replace(' ', '_')}")
            st.caption(
                "Gap (pp): positive-close rate on signal days minus all other days, "
                "in percentage points. The in/out-of-sample split is at the sample "
                "midpoint by observation count. Net-of-cost return assumes one "
                f"round trip at {cost_bps} bps one-way per contiguous signal block. "
                "A 'Validated' assessment requires FDR p < 0.05, an out-of-sample "
                "gap with the same sign as in-sample, and at least half the "
                "in-sample magnitude. The net-positive flag is assessed independently."
            )

ui.banner(
    "info",
    "For most well-studied assets, <b>zero validated patterns is the statistically "
    "correct and expected outcome</b>. Calendar effects documented in the academic "
    "literature have largely attenuated or disappeared following publication.",
)

ui.footer_disclaimer()
