"""
Calendar Anomaly Lab — tests whether calendar patterns predict positive-return days.
Primary metric: green-day rate (proportion of days with return > 0).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import anomalies as an
from src import data as dt
from src.theme import PRIMARY, BENCHMARK, NEUTRAL, apply_chart_theme


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()

today = date.today()
start_date = st.sidebar.date_input(
    "Start date",
    value=date(today.year - 15, today.month, today.day),
    min_value=date(2000, 1, 1),
    max_value=today - timedelta(days=365),
    help="15 years gives ~3,750 observations — the minimum for reasonable statistical power.",
)
end_date = today

cost_bps = st.sidebar.slider(
    "Transaction cost (bps, one-way)",
    min_value=1, max_value=20, value=5, step=1,
    help=(
        "Used to compute post-cost mean return on signal days. "
        "Does not affect the primary green-rate hypothesis test."
    ),
)

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    return dt.get_prices(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_lab(
    prices: pd.Series, cost_bps: float
) -> list[an.AnomalyCategory]:
    return an.run_anomaly_lab(prices, cost_bps=cost_bps)


# ── Fetch prices ──────────────────────────────────────────────────────────────

with st.spinner(f"Loading {ticker} price data…"):
    try:
        price_df = _fetch_prices(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        st.stop()

if price_df.empty or "adj_close" not in price_df.columns:
    st.error(f"No price data for **{ticker}**. Check the ticker symbol.")
    st.stop()

prices = price_df["adj_close"]
n_days = len(prices) - 1   # trading days after pct_change

if n_days < 500:
    st.warning(
        f"Only {n_days} trading days available — extend the date range for "
        "reliable statistical estimates."
    )

# ── Run lab ───────────────────────────────────────────────────────────────────

with st.spinner("Running hypothesis tests…"):
    categories = _run_lab(prices, float(cost_bps))

all_hyps = [h for cat in categories for h in cat.hypotheses]

# ── Summary counts ────────────────────────────────────────────────────────────

n_total    = len(all_hyps)
n_raw_sig  = sum(1 for h in all_hyps if not np.isnan(h.p_raw)  and h.p_raw  < 0.05)
n_fdr_sig  = sum(1 for h in all_hyps if not np.isnan(h.p_fdr)  and h.p_fdr  < 0.05)
n_real     = sum(1 for h in all_hyps if h.verdict == "Real pattern")
n_tradable = sum(1 for h in all_hyps if h.tradable)

baseline_green = float(prices.pct_change().dropna().gt(0).mean())

# ── Title + intro ─────────────────────────────────────────────────────────────

st.title("Calendar Anomaly Lab")

st.markdown(
    f"**{ticker}** · {start_date} – {end_date} · {n_days:,} trading days"
)

st.markdown("""
**The multiple-testing problem**

Testing enough calendar hypotheses guarantees that some will appear statistically
significant purely by chance. With 20 tests at α = 0.05, one false positive is
expected under the null hypothesis of no effect. The Benjamini-Hochberg
false-discovery-rate correction adjusts p-values upward to account for the number
of simultaneous tests. This page applies that correction across all 20 hypotheses
in a single pass.

One important caveat: the stated test count still understates the true
data-mining burden. The window definitions — for example, ±3 trading days around
the new moon, or the last trading day of a month plus the first three of the next —
reflect choices made after reviewing the academic literature. Each such design
decision introduces hidden degrees of freedom that formal multiple-testing
corrections cannot account for.
""")

# ── Summary banner ────────────────────────────────────────────────────────────

st.info(
    f"**{n_total} calendar patterns tested.** "
    f"**{n_raw_sig}** appeared significant at the raw p < 0.05 threshold.  "
    f"**{n_fdr_sig}** survived the Benjamini-Hochberg multiple-testing correction (FDR q = 0.05).  "
    f"**{n_real}** also replicated out-of-sample with at least half the in-sample effect size.  "
    f"Of those, **{n_tradable}** showed a positive mean return after transaction costs.  "
    f"Baseline positive-return rate across all trading days: **{baseline_green*100:.1f}%**."
)

# ── Rendering helpers ─────────────────────────────────────────────────────────

def _category_badge_html(cat: an.AnomalyCategory) -> str:
    n_real_cat = sum(1 for h in cat.hypotheses if h.verdict == "Real pattern")
    n_tot_cat  = len(cat.hypotheses)
    if n_real_cat == 0:
        style = f"background:{NEUTRAL};color:white"
        text  = "All noise"
    else:
        style = f"background:{BENCHMARK};color:white"
        text  = f"{n_real_cat} of {n_tot_cat} — real pattern"
    return (
        f'<span style="{style};padding:3px 10px;border-radius:4px;'
        f'font-size:0.85em;font-weight:bold">{text}</span>'
    )


def _green_rate_chart(
    bucket_stats: list[an.BucketStats],
    signal_labels: set[str],
    baseline: float,
) -> go.Figure:
    labels      = [b.label for b in bucket_stats]
    green_rates = [b.green_rate * 100 for b in bucket_stats]
    se_vals     = [b.se_green * 100 for b in bucket_stats]
    colors      = [PRIMARY if lbl in signal_labels else NEUTRAL for lbl in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=green_rates,
        error_y=dict(type="data", array=se_vals, visible=True,
                     color=NEUTRAL, thickness=1.5, width=4),
        marker_color=colors,
        text=[f"{g:.1f}%" for g in green_rates],
        textposition="outside",
        hovertemplate="%{x}<br>Green-day rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=baseline * 100,
        line_dash="dash", line_color=BENCHMARK, line_width=1.5,
        annotation_text=f"Baseline {baseline*100:.1f}%",
        annotation_position="top right",
        annotation_font_size=11,
    )
    lo = max(0,   min(green_rates) - 4)
    hi = min(100, max(green_rates) + 6)
    fig.update_layout(
        yaxis_title="Green-day rate (%)",
        yaxis=dict(range=[lo, hi]),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


def _hypothesis_df(hyps: list[an.HypothesisResult]) -> pd.DataFrame:
    rows = []
    for h in hyps:
        rows.append({
            "Signal":            h.signal_label,
            "N (signal)":        h.n_signal,
            "Green % signal":    h.green_rate_signal,
            "Green % other":     h.green_rate_other,
            "Gap (pp)":          h.green_rate_gap * 100 if not np.isnan(h.green_rate_gap) else float("nan"),
            "Z-stat":            h.z_stat,
            "Raw p":             h.p_raw,
            "FDR p":             h.p_fdr,
            "IS gap (pp)":       h.is_green_rate_gap  * 100 if not np.isnan(h.is_green_rate_gap)  else float("nan"),
            "OOS gap (pp)":      h.oos_green_rate_gap * 100 if not np.isnan(h.oos_green_rate_gap) else float("nan"),
            "Mean return (bps)": h.mean_return_bps,
            "Post-cost (bps)":   h.post_cost_mean_return_bps,
            "Verdict":           h.verdict,
            "Tradable?":         "Yes" if h.tradable else "No",
        })
    return pd.DataFrame(rows)


_TABLE_CFG = {
    "Signal":            st.column_config.TextColumn("Signal"),
    "N (signal)":        st.column_config.NumberColumn("N",              format="%d"),
    "Green % signal":    st.column_config.NumberColumn("Green % signal", format="%.1f%%"),
    "Green % other":     st.column_config.NumberColumn("Green % other",  format="%.1f%%"),
    "Gap (pp)":          st.column_config.NumberColumn("Gap (pp)",       format="%+.2f",
                             help="Green-rate on signal days minus other days, in percentage points"),
    "Z-stat":            st.column_config.NumberColumn("Z-stat",         format="%.2f"),
    "Raw p":             st.column_config.NumberColumn("Raw p",          format="%.3f"),
    "FDR p":             st.column_config.NumberColumn("FDR p",          format="%.3f",
                             help="Benjamini-Hochberg corrected p-value across all 20 tests"),
    "IS gap (pp)":       st.column_config.NumberColumn("IS gap (pp)",    format="%+.2f",
                             help="Green-rate gap in the first half of the sample"),
    "OOS gap (pp)":      st.column_config.NumberColumn("OOS gap (pp)",   format="%+.2f",
                             help="Green-rate gap in the second half of the sample"),
    "Mean return (bps)": st.column_config.NumberColumn("Mean ret (bps)", format="%.1f",
                             help="Mean daily return on signal days in basis points"),
    "Post-cost (bps)":   st.column_config.NumberColumn("Post-cost (bps)",format="%.1f",
                             help="Mean return after round-trip transaction costs"),
    "Verdict":           st.column_config.TextColumn("Verdict"),
    "Tradable?":         st.column_config.TextColumn("Tradable?"),
}


# ── Per-category sections ─────────────────────────────────────────────────────

for cat in categories:
    st.header(cat.name)
    st.markdown(
        _category_badge_html(cat) + f"&nbsp;&nbsp;{cat.description}",
        unsafe_allow_html=True,
    )

    signal_labels = {h.signal_label for h in cat.hypotheses}

    st.plotly_chart(
        _green_rate_chart(cat.bucket_stats, signal_labels, baseline_green),
        use_container_width=True,
    )

    with st.expander("Detailed statistics"):
        df = _hypothesis_df(cat.hypotheses)
        if df.empty:
            st.info("No testable signal buckets (insufficient observations).")
        else:
            st.dataframe(
                df,
                column_config=_TABLE_CFG,
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "Gap (pp): positive-return rate on signal days minus all other days, "
            "expressed in percentage points. "
            "The IS/OOS split is at the sample midpoint by observation count. "
            f"Post-cost return assumes one round-trip at {cost_bps} bps one-way "
            "per contiguous signal-day block. "
            "A verdict of 'Real pattern' requires FDR p < 0.05, out-of-sample gap "
            "in the same direction as the in-sample gap, and |OOS| ≥ 0.5 × |IS|. "
            "The 'Tradable?' flag is assessed independently: post-cost mean return > 0."
        )

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.markdown(
    "_This tool is designed as a skepticism aid, not a signal generator. "
    "For most well-studied assets, zero patterns surviving all three criteria "
    "is the statistically correct and expected outcome. "
    "Calendar effects that appeared significant in the academic literature have "
    "largely attenuated or disappeared following publication._"
)
