"""Portfolio Construction — mean-variance and risk-parity allocation with robust estimators."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import data
from src import estimators as est
from src import portfolio as pf
from src.theme import PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Portfolio & Risk", "Portfolio Construction",
    "Long-only allocation by maximum Sharpe ratio, minimum variance, or risk "
    "parity, with shrinkage estimators for the inputs, optional Michaud "
    "resampling, and the mean-variance efficient frontier.",
)

METHOD_LABELS = {
    "max_sharpe":   "Maximum Sharpe Ratio",
    "min_variance": "Minimum Variance",
    "risk_parity":  "Risk Parity",
}
COV_LABELS = {
    "ledoit_wolf": "Ledoit-Wolf shrinkage",
    "oas":         "Oracle Approximating Shrinkage",
    "sample":      "Sample (unreliable on short windows)",
}
MEAN_LABELS = {
    "james_stein": "James-Stein (data-driven shrinkage)",
    "sample":      "Sample mean",
}

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with st.form("construction_params"):
    st.markdown('<p class="qrt-kicker">Mandate Parameters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        tickers = ui.ticker_list_input(
            "Investment Universe", "AAPL\nMSFT\nGOOGL\nSPY\nTLT\nGLD", height=150)
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            start_date, end_date = ui.date_range_input(
                "Estimation Window", today - timedelta(days=365 * 5), today)
            method = st.radio("Objective", list(METHOD_LABELS),
                              format_func=METHOD_LABELS.get)
        with cc2:
            rf = ui.rf_rate_input()
            weight_cap_pct = st.slider(
                "Single-Position Limit (%)", min_value=10, max_value=100, value=100, step=5,
                help="Maximum allocation to any single asset. Not applied to risk "
                     "parity, which is unconstrained by construction.",
            )

    st.markdown('<p class="qrt-kicker" style="margin-top:0.6rem">Estimation Methodology</p>',
                unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1.2, 1.2, 1.6])
    with e1:
        cov_estimator = st.selectbox("Covariance Estimator", list(COV_LABELS),
                                     format_func=COV_LABELS.get)
    with e2:
        mean_estimator = st.selectbox("Expected-Return Estimator", list(MEAN_LABELS),
                                      format_func=MEAN_LABELS.get)
    with e3:
        use_resampling = st.checkbox(
            "Michaud Resampling (200 bootstraps; slow first run)", value=False,
            help="Bootstraps the return history 200 times, re-optimizes each "
                 "sample, and averages the weight vectors — reducing sensitivity "
                 "to estimation error (Michaud, 1998).",
        )

    submitted = st.form_submit_button("Run Construction", type="primary")

weight_cap = weight_cap_pct / 100 if method != "risk_parity" else 1.0

# ── Validation ────────────────────────────────────────────────────────────────

if len(tickers) < 2:
    ui.banner("warn", "Specify at least two instruments in the investment universe.")
    st.stop()

if method != "risk_parity" and weight_cap * len(tickers) < 1.0:
    ui.banner(
        "warn",
        f"A {weight_cap_pct}% single-position limit across {len(tickers)} instruments "
        f"caps total allocation at {weight_cap_pct * len(tickers):.0f}%, below 100%. "
        "No feasible portfolio exists — raise the limit or expand the universe.",
    )
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(ticker: str, start: date, end: date) -> pd.Series:
    df = data.get_prices(ticker, start, end)
    return df["adj_close"] if not df.empty else pd.Series(dtype=float, name=ticker)


with st.spinner("Retrieving price histories..."):
    raw_series = {t: _fetch(t, start_date, end_date) for t in tickers}

failed = [t for t, s in raw_series.items() if s.empty]
if failed:
    ui.banner("warn", f"No data for: <span class='mono'>{', '.join(failed)}</span> — excluded.")

price_df = pd.DataFrame({t: s for t, s in raw_series.items() if not s.empty}).dropna()

if price_df.shape[1] < 2:
    ui.banner("error", "At least two instruments with overlapping history are required.")
    st.stop()

active_tickers = list(price_df.columns)
returns_df = price_df.pct_change().dropna()

# ── Estimates ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_estimates(returns: pd.DataFrame, cov_est: str, mean_est: str):
    mu  = est.MEAN_ESTIMATORS[mean_est](returns)
    cov = est.COV_ESTIMATORS[cov_est](returns)
    return mu, cov


mu, cov = _compute_estimates(returns_df, cov_estimator, mean_estimator)
asset_vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)

# ── Universe overview ─────────────────────────────────────────────────────────

with ui.panel("Universe Overview"):
    stats_df = pd.DataFrame({
        "Expected Return": (mu * 100).map("{:.1f}%".format),
        "Volatility":      (asset_vols * 100).map("{:.1f}%".format),
        "Sharpe":          ((mu - rf) / asset_vols).map("{:.2f}".format),
    })

    corr = returns_df.corr()
    corr_fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    corr_fig.update_layout(title="Return correlation", height=350,
                           margin=dict(l=10, r=10, t=40, b=10))
    apply_chart_theme(corr_fig)

    col_stats, col_corr = st.columns([1, 2])
    with col_stats:
        st.dataframe(stats_df, width="stretch")
    with col_corr:
        st.plotly_chart(corr_fig, width="stretch", config=CHART_CONFIG)

# ── Optimization ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _optimize(mu, cov, returns, method_key, rf, weight_cap,
              cov_estimator, mean_estimator, use_resampling):
    equal_w = pd.Series(1.0 / len(mu), index=mu.index)
    equal_  = pf.portfolio_stats(equal_w, mu, cov, rf, method="equal_weight")

    if method_key == "max_sharpe":
        single_opt = pf.max_sharpe(mu, cov, rf=rf, weight_cap=weight_cap)
        tangency   = single_opt
    elif method_key == "min_variance":
        single_opt = pf.min_variance(mu, cov, rf=rf, weight_cap=weight_cap)
        tangency   = pf.max_sharpe(mu, cov, rf=rf, weight_cap=weight_cap)
    else:
        single_opt = pf.risk_parity(mu, cov, rf=rf)
        tangency   = pf.max_sharpe(mu, cov, rf=rf, weight_cap=1.0)

    if use_resampling:
        r_weights = est.resampled_weights(
            returns, method_key, rf, weight_cap, cov_estimator, mean_estimator)
        opt = pf.portfolio_stats(r_weights, mu, cov, rf, method=f"{method_key}_resampled")
    else:
        opt = single_opt

    frontier = pf.efficient_frontier(mu, cov, n_points=40, weight_cap=weight_cap)
    return opt, equal_, tangency, frontier


spinner_msg = (
    "Optimizing — resampling 200 bootstraps, first run approximately 30 seconds..."
    if use_resampling
    else "Optimizing — first run approximately 10 seconds for the frontier..."
)
try:
    with st.spinner(spinner_msg):
        opt_result, equal_result, tangency, frontier_df = _optimize(
            mu, cov, returns_df, method, rf, weight_cap,
            cov_estimator, mean_estimator, use_resampling,
        )
except Exception as exc:
    ui.banner("error", f"Optimization failed: {exc}")
    st.stop()

# Persist for the Risk Analytics page
st.session_state["portfolio_weights"] = opt_result.weights
st.session_state["portfolio_prices"]  = price_df
st.session_state["portfolio_returns"] = returns_df
st.session_state["portfolio_method"]  = method + ("_resampled" if use_resampling else "")
st.session_state["portfolio_cov"]     = cov

# ── Target allocation ─────────────────────────────────────────────────────────

label_suffix = " (Michaud resampled)" if use_resampling else ""
ui.section(f"Target Allocation — {METHOD_LABELS[method]}{label_suffix}")

ui.kpi_row([
    {"label": "Expected Return", "value": f"{opt_result.expected_return * 100:.2f}%",
     "delta": f"{(opt_result.expected_return - equal_result.expected_return) * 100:+.2f}% vs equal weight"},
    {"label": "Volatility", "value": f"{opt_result.volatility * 100:.2f}%",
     "delta": f"{(opt_result.volatility - equal_result.volatility) * 100:+.2f}% vs equal weight",
     "delta_kind": "pos" if opt_result.volatility <= equal_result.volatility else "neg"},
    {"label": "Sharpe Ratio", "value": f"{opt_result.sharpe:.3f}",
     "delta": f"{opt_result.sharpe - equal_result.sharpe:+.3f} vs equal weight"},
    {"label": "Equal-Weight Return", "value": f"{equal_result.expected_return * 100:.2f}%"},
    {"label": "Equal-Weight Sharpe", "value": f"{equal_result.sharpe:.3f}"},
])

# Risk contribution
Sigma    = cov.values
w_vec    = opt_result.weights.reindex(active_tickers).fillna(0.0).values
port_var = float(w_vec @ Sigma @ w_vec)
rc_pct   = pd.Series(w_vec * (Sigma @ w_vec) / port_var * 100, index=active_tickers)

sorted_idx  = opt_result.weights.sort_values(ascending=True).index
weight_vals = opt_result.weights.reindex(sorted_idx).values * 100
rc_vals     = rc_pct.reindex(sorted_idx).values
yticks      = sorted_idx.tolist()
chart_h     = max(300, 52 * len(active_tickers))

bar_fig = go.Figure(go.Bar(
    x=weight_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in weight_vals], textposition="outside",
    marker_color=PRIMARY,
))
bar_fig.update_layout(
    title="Capital allocation (%)", xaxis_title="Weight (%)",
    height=chart_h, margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(weight_vals) * 1.25]),
)
apply_chart_theme(bar_fig)

rc_fig = go.Figure(go.Bar(
    x=rc_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in rc_vals], textposition="outside",
    marker_color=NEGATIVE,
))
rc_fig.update_layout(
    title="Risk contribution (% of portfolio variance)", xaxis_title="Risk contribution (%)",
    height=chart_h, margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(rc_vals) * 1.25]),
)
apply_chart_theme(rc_fig)

col_bar, col_rc = st.columns(2)
with col_bar:
    with ui.panel():
        st.plotly_chart(bar_fig, width="stretch", config=CHART_CONFIG)
with col_rc:
    with ui.panel():
        st.plotly_chart(rc_fig, width="stretch", config=CHART_CONFIG)

# ── Efficient frontier ────────────────────────────────────────────────────────

with ui.panel("Efficient Frontier"):
    ef_fig = go.Figure()

    if not frontier_df.empty:
        ef_fig.add_trace(go.Scatter(
            x=frontier_df["volatility"] * 100,
            y=frontier_df["expected_return"] * 100,
            mode="lines", name="Efficient frontier",
            line=dict(color=PRIMARY, width=2.5),
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))
        cml_max = frontier_df["volatility"].max() * 100 * 1.3
        ef_fig.add_trace(go.Scatter(
            x=[0.0, cml_max],
            y=[rf * 100, rf * 100 + tangency.sharpe * cml_max],
            mode="lines", name=f"Capital market line (Sharpe {tangency.sharpe:.2f})",
            line=dict(color=BENCHMARK, width=1.5, dash="dash"),
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

    ef_fig.add_trace(go.Scatter(
        x=(asset_vols * 100).tolist(), y=(mu * 100).tolist(),
        mode="markers+text", name="Individual assets",
        marker=dict(size=9, color=NEUTRAL, symbol="circle"),
        text=active_tickers, textposition="top center",
        hovertemplate="%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))
    ef_fig.add_trace(go.Scatter(
        x=[equal_result.volatility * 100], y=[equal_result.expected_return * 100],
        mode="markers+text", name="Equal weight",
        marker=dict(size=12, color=POSITIVE, symbol="diamond"),
        text=["EW"], textposition="top right",
        hovertemplate=(f"Equal weight<br>Vol: {equal_result.volatility * 100:.1f}%<br>"
                       f"Return: {equal_result.expected_return * 100:.1f}%<br>"
                       f"Sharpe: {equal_result.sharpe:.2f}<extra></extra>"),
    ))
    ef_fig.add_trace(go.Scatter(
        x=[opt_result.volatility * 100], y=[opt_result.expected_return * 100],
        mode="markers+text", name=METHOD_LABELS[method] + label_suffix,
        marker=dict(size=14, color=NEGATIVE, symbol="star"),
        text=["Target"], textposition="top right",
        hovertemplate=(f"{METHOD_LABELS[method]}<br>Vol: {opt_result.volatility * 100:.1f}%<br>"
                       f"Return: {opt_result.expected_return * 100:.1f}%<br>"
                       f"Sharpe: {opt_result.sharpe:.2f}<extra></extra>"),
    ))

    ef_fig.update_layout(
        xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(ef_fig)
    st.plotly_chart(ef_fig, width="stretch", config=CHART_CONFIG)

# ── In-sample performance ─────────────────────────────────────────────────────

with ui.panel("In-Sample Performance — Target Allocation vs Equal Weight"):
    w_aligned = opt_result.weights.reindex(returns_df.columns).fillna(0.0)
    port_rets = returns_df @ w_aligned.values
    ew_rets   = returns_df.mean(axis=1)

    wealth_opt = (1 + port_rets).cumprod()
    wealth_ew  = (1 + ew_rets).cumprod()

    perf_fig = go.Figure()
    perf_fig.add_trace(go.Scatter(
        x=wealth_opt.index, y=wealth_opt.values, mode="lines",
        name="Target allocation", line=dict(color=PRIMARY, width=2),
    ))
    perf_fig.add_trace(go.Scatter(
        x=wealth_ew.index, y=wealth_ew.values, mode="lines",
        name="Equal weight", line=dict(color=POSITIVE, width=1.5, dash="dash"),
    ))
    perf_fig.update_layout(
        yaxis_title="Growth of $1 (log)", yaxis_type="log",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(perf_fig)
    st.plotly_chart(perf_fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Realized growth of $1 holding the target weights with daily rebalancing, "
        "gross of transaction costs, over the same window the weights were "
        "estimated on. This is an in-sample illustration, not a forecast — for "
        "out-of-sample evidence use Strategy Simulation's walk-forward optimizer, "
        "which re-fits the allocation at each rebalance."
    )

ui.banner(
    "info",
    "The target allocation has been staged for the <b>Risk Analytics</b> tool — "
    "open it from the home screen to evaluate Value at Risk, factor exposure, "
    "and stress scenarios for this portfolio.",
)

# ── Allocation detail ─────────────────────────────────────────────────────────

with st.expander("Allocation Detail and Export"):
    detail_df = pd.DataFrame({
        "Weight":            opt_result.weights.map(lambda x: f"{x * 100:.2f}%"),
        "Risk Contribution": rc_pct.reindex(opt_result.weights.index).map(lambda x: f"{x:.1f}%"),
    })
    st.dataframe(detail_df, width="stretch")

    csv_bytes = pd.DataFrame({
        "ticker": opt_result.weights.index,
        "weight": opt_result.weights.values,
    }).to_csv(index=False)
    st.download_button("Export Allocation (CSV)", csv_bytes,
                       file_name=f"allocation_{method}.csv", mime="text/csv")

ui.footer_disclaimer()
