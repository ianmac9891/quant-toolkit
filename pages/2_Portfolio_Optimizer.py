"""Portfolio optimizer: max Sharpe, min variance, risk parity."""

from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data
from src import estimators as est
from src import portfolio as pf
from src.theme import PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, apply_chart_theme

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

METHOD_LABELS = {
    "max_sharpe":   "Max Sharpe",
    "min_variance": "Min Variance",
    "risk_parity":  "Risk Parity",
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Inputs")

raw_tickers = st.sidebar.text_area(
    "Tickers",
    "AAPL\nMSFT\nGOOGL\nSPY\nTLT\nGLD",
    height=130,
    help="Enter one ticker per line or separate them with commas.",
)
tickers = sorted(set(t for t in re.split(r"[\s,]+", raw_tickers.strip().upper()) if t))

today = date.today()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(today - timedelta(days=365 * 5), today),
    min_value=date(1990, 1, 1),
    max_value=today,
)

rf_pct = st.sidebar.number_input(
    "Risk-free rate (% annual)", min_value=0.0, max_value=20.0, value=4.5, step=0.25,
    help="Used to compute the Sharpe ratio and to draw the Capital Market Line. A common proxy is the current 3-month T-bill yield.",
)
rf = rf_pct / 100

method = st.sidebar.radio(
    "Optimization method",
    list(METHOD_LABELS),
    format_func=METHOD_LABELS.get,
)

weight_cap_pct = st.sidebar.slider(
    "Max weight per asset (%)",
    min_value=10, max_value=100, value=100, step=5,
    disabled=(method == "risk_parity"),
    help="Maximum allocation to any single asset. Disabled for Risk Parity, which is unconstrained by design.",
)
weight_cap = weight_cap_pct / 100 if method != "risk_parity" else 1.0

# Robustness controls
COV_LABELS = {
    "ledoit_wolf": "Ledoit-Wolf shrinkage",
    "oas":         "OAS shrinkage",
    "sample":      "Sample (unreliable on short windows)",
}
MEAN_LABELS = {
    "james_stein": "James-Stein (shrunk)",
    "sample":      "Sample mean",
}
with st.sidebar.expander("Robustness", expanded=True):
    cov_estimator = st.selectbox(
        "Covariance estimator", list(COV_LABELS), format_func=COV_LABELS.get,
    )
    mean_estimator = st.selectbox(
        "Return estimator", list(MEAN_LABELS), format_func=MEAN_LABELS.get,
    )
    use_resampling = st.checkbox(
        "Michaud resampling (200 bootstraps, slow first run)",
        value=False,
        help=(
            "Bootstraps the return series 200 times, re-optimizes on each sample, "
            "and averages the resulting weight vectors. This reduces sensitivity to "
            "estimation error in the inputs (Michaud, 1998)."
        ),
    )

# ─── Validation ───────────────────────────────────────────────────────────────

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers.")
    st.stop()

if method != "risk_parity" and weight_cap * len(tickers) < 1.0:
    st.warning(
        f"The per-asset cap of {weight_cap_pct}% × {len(tickers)} tickers "
        f"sums to {weight_cap_pct * len(tickers):.0f}%, which is below 100%. "
        "No feasible portfolio exists under these constraints. "
        "Increase the cap or add more tickers to proceed."
    )
    st.stop()

# ─── Data fetch ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(ticker: str, start: date, end: date) -> pd.Series:
    df = data.get_prices(ticker, start, end)
    return df["adj_close"] if not df.empty else pd.Series(dtype=float, name=ticker)


with st.spinner("Fetching price data..."):
    raw_series = {t: _fetch(t, start_date, end_date) for t in tickers}

failed = [t for t, s in raw_series.items() if s.empty]
if failed:
    st.warning(f"No data for: {', '.join(failed)} — skipping.")

price_df = pd.DataFrame({t: s for t, s in raw_series.items() if not s.empty}).dropna()

if price_df.shape[1] < 2:
    st.error("Need at least 2 tickers with overlapping data in the selected date range.")
    st.stop()

active_tickers = list(price_df.columns)
returns_df = price_df.pct_change().dropna()

# ─── Estimates ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_estimates(
    returns: pd.DataFrame, cov_est: str, mean_est: str
) -> tuple[pd.Series, pd.DataFrame]:
    mu  = est.MEAN_ESTIMATORS[mean_est](returns)
    cov = est.COV_ESTIMATORS[cov_est](returns)
    return mu, cov


mu, cov = _compute_estimates(returns_df, cov_estimator, mean_estimator)
asset_vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)

# ─── Section 1: Asset overview ────────────────────────────────────────────────

st.header("Asset overview")

stats_df = pd.DataFrame({
    "Ann. return": (mu * 100).map("{:.1f}%".format),
    "Ann. vol":    (asset_vols * 100).map("{:.1f}%".format),
    "Sharpe":      ((mu - rf) / asset_vols).map("{:.2f}".format),
})

corr = returns_df.corr()
corr_fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
))
corr_fig.update_layout(
    title="Return correlation",
    height=350,
    margin=dict(l=10, r=10, t=40, b=10),
)
apply_chart_theme(corr_fig)

col_stats, col_corr = st.columns([1, 2])
with col_stats:
    st.dataframe(stats_df, use_container_width=True)
with col_corr:
    st.plotly_chart(corr_fig, use_container_width=True)

# ─── Optimization ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    returns: pd.DataFrame,
    method_key: str,
    rf: float,
    weight_cap: float,
    cov_estimator: str,
    mean_estimator: str,
    use_resampling: bool,
) -> tuple:
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
            returns, method_key, rf, weight_cap, cov_estimator, mean_estimator,
        )
        opt = pf.portfolio_stats(
            r_weights, mu, cov, rf, method=f"{method_key}_resampled"
        )
    else:
        opt = single_opt

    frontier = pf.efficient_frontier(mu, cov, n_points=40, weight_cap=weight_cap)
    return opt, equal_, tangency, frontier


spinner_msg = (
    "Optimizing — resampling 200 bootstraps, first run ~30 s..."
    if use_resampling
    else "Optimizing portfolio (first run ~10 s for the frontier)..."
)
try:
    with st.spinner(spinner_msg):
        opt_result, equal_result, tangency, frontier_df = _optimize(
            mu, cov, returns_df, method, rf, weight_cap,
            cov_estimator, mean_estimator, use_resampling,
        )
except Exception as exc:
    st.error(f"Optimization failed: {exc}")
    st.stop()

# Persist for the Risk Model page
st.session_state["risk_model_weights"] = opt_result.weights
st.session_state["risk_model_prices"]  = price_df
st.session_state["risk_model_returns"] = returns_df
st.session_state["risk_model_method"]  = method
st.session_state["risk_model_cov"]     = cov

# ─── Section 2: Optimal weights ───────────────────────────────────────────────

label_suffix = " (Michaud resampled)" if use_resampling else ""
st.header(f"Optimal weights — {METHOD_LABELS[method]}{label_suffix}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(
    "Opt. return", f"{opt_result.expected_return * 100:.2f}%",
    delta=f"{(opt_result.expected_return - equal_result.expected_return) * 100:+.2f}% vs EW",
)
c2.metric(
    "Opt. vol", f"{opt_result.volatility * 100:.2f}%",
    delta=f"{(opt_result.volatility - equal_result.volatility) * 100:+.2f}% vs EW",
    delta_color="inverse",
)
c3.metric(
    "Opt. Sharpe", f"{opt_result.sharpe:.3f}",
    delta=f"{opt_result.sharpe - equal_result.sharpe:+.3f} vs EW",
)
c4.metric("EW return", f"{equal_result.expected_return * 100:.2f}%")
c5.metric("EW vol",    f"{equal_result.volatility * 100:.2f}%")
c6.metric("EW Sharpe", f"{equal_result.sharpe:.3f}")

# Risk contribution
Sigma   = cov.values
w_vec   = opt_result.weights.reindex(active_tickers).fillna(0.0).values
port_var = float(w_vec @ Sigma @ w_vec)
rc_pct   = pd.Series(
    w_vec * (Sigma @ w_vec) / port_var * 100,
    index=active_tickers,
)

sorted_idx   = opt_result.weights.sort_values(ascending=True).index
weight_vals  = opt_result.weights.reindex(sorted_idx).values * 100
rc_vals      = rc_pct.reindex(sorted_idx).values
yticks       = sorted_idx.tolist()
chart_h      = max(300, 52 * len(active_tickers))

bar_fig = go.Figure(go.Bar(
    x=weight_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in weight_vals],
    textposition="outside",
    marker_color=PRIMARY,
))
bar_fig.update_layout(
    title="Capital allocation (%)",
    xaxis_title="Weight (%)",
    height=chart_h,
    margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(weight_vals) * 1.25]),
)
apply_chart_theme(bar_fig)

rc_fig = go.Figure(go.Bar(
    x=rc_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in rc_vals],
    textposition="outside",
    marker_color=NEGATIVE,
))
rc_fig.update_layout(
    title="Risk contribution (% of portfolio vol)",
    xaxis_title="Risk contribution (%)",
    height=chart_h,
    margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(rc_vals) * 1.25]),
)
apply_chart_theme(rc_fig)

col_bar, col_rc = st.columns(2)
with col_bar:
    st.plotly_chart(bar_fig, use_container_width=True)
with col_rc:
    st.plotly_chart(rc_fig, use_container_width=True)

# ─── Section 3: Efficient frontier ────────────────────────────────────────────

st.header("Efficient frontier")

ef_fig = go.Figure()

if not frontier_df.empty:
    ef_fig.add_trace(go.Scatter(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["expected_return"] * 100,
        mode="lines",
        name="Efficient frontier",
        line=dict(color=PRIMARY, width=2.5),
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

if not frontier_df.empty:
    cml_max = frontier_df["volatility"].max() * 100 * 1.3
    ef_fig.add_trace(go.Scatter(
        x=[0.0, cml_max],
        y=[rf * 100, rf * 100 + tangency.sharpe * cml_max],
        mode="lines",
        name=f"CML  (Sharpe = {tangency.sharpe:.2f})",
        line=dict(color=BENCHMARK, width=1.5, dash="dash"),
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

ef_fig.add_trace(go.Scatter(
    x=(asset_vols * 100).tolist(),
    y=(mu * 100).tolist(),
    mode="markers+text",
    name="Individual assets",
    marker=dict(size=9, color=NEUTRAL, symbol="circle"),
    text=active_tickers,
    textposition="top center",
    hovertemplate="%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
))

ef_fig.add_trace(go.Scatter(
    x=[equal_result.volatility * 100],
    y=[equal_result.expected_return * 100],
    mode="markers+text",
    name="Equal weight",
    marker=dict(size=12, color=POSITIVE, symbol="diamond"),
    text=["EW"],
    textposition="top right",
    hovertemplate=(
        f"Equal weight<br>Vol: {equal_result.volatility * 100:.1f}%<br>"
        f"Return: {equal_result.expected_return * 100:.1f}%<br>"
        f"Sharpe: {equal_result.sharpe:.2f}<extra></extra>"
    ),
))

ef_fig.add_trace(go.Scatter(
    x=[opt_result.volatility * 100],
    y=[opt_result.expected_return * 100],
    mode="markers+text",
    name=METHOD_LABELS[method] + label_suffix,
    marker=dict(size=14, color=NEGATIVE, symbol="star"),
    text=[METHOD_LABELS[method]],
    textposition="top right",
    hovertemplate=(
        f"{METHOD_LABELS[method]}<br>Vol: {opt_result.volatility * 100:.1f}%<br>"
        f"Return: {opt_result.expected_return * 100:.1f}%<br>"
        f"Sharpe: {opt_result.sharpe:.2f}<extra></extra>"
    ),
))

ef_fig.update_layout(
    xaxis_title="Annualized volatility (%)",
    yaxis_title="Annualized return (%)",
    height=520,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="closest",
    legend=dict(x=0.02, y=0.98),
)
apply_chart_theme(ef_fig)
st.plotly_chart(ef_fig, use_container_width=True)

# ─── Section 4: Weights detail ────────────────────────────────────────────────

with st.expander("Weights and risk breakdown"):
    detail_df = pd.DataFrame({
        "Weight":            opt_result.weights.map(lambda x: f"{x * 100:.2f}%"),
        "Risk contribution": rc_pct.reindex(opt_result.weights.index).map(
            lambda x: f"{x:.1f}%"
        ),
    })
    st.dataframe(detail_df, use_container_width=True)

    csv_bytes = pd.DataFrame({
        "ticker": opt_result.weights.index,
        "weight": opt_result.weights.values,
    }).to_csv(index=False)
    st.download_button(
        "Download weights (CSV)",
        csv_bytes,
        file_name=f"portfolio_{method}.csv",
        mime="text/csv",
    )
