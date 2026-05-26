"""Risk model: VaR/CVaR, Monte Carlo simulation, factor exposure, stress tests."""

from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data, portfolio as pf, risk
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL,
    PRIMARY_10, PRIMARY_18,
    apply_chart_theme,
)

st.set_page_config(page_title="Risk Model", layout="wide")

st.title("Risk Model")

_METHOD_NAMES = {
    "max_sharpe":              "Max Sharpe",
    "min_variance":            "Min Variance",
    "risk_parity":             "Risk Parity",
    "max_sharpe_resampled":    "Max Sharpe (Michaud resampled)",
    "min_variance_resampled":  "Min Variance (Michaud resampled)",
    "risk_parity_resampled":   "Risk Parity (Michaud resampled)",
}

# ─── Input ────────────────────────────────────────────────────────────────────

input_mode = st.radio(
    "Portfolio source",
    ["Load from optimizer", "Enter manually"],
    horizontal=True,
)

if input_mode == "Load from optimizer":
    if "risk_model_weights" not in st.session_state:
        st.info(
            "No portfolio found in the current session. Navigate to the "
            "**Portfolio Optimizer** page to construct a portfolio, then return here."
        )
        st.stop()

    weights     = st.session_state["risk_model_weights"]
    price_df    = st.session_state["risk_model_prices"]
    returns_df  = st.session_state["risk_model_returns"]
    method_used = st.session_state.get("risk_model_method", "")
    _cov_stored = st.session_state.get("risk_model_cov")
    cov         = _cov_stored if _cov_stored is not None else pf.covariance_matrix(returns_df)

else:
    st.sidebar.header("Manual portfolio")

    raw_manual = st.sidebar.text_area(
        "Ticker  weight  (one per line)",
        "AAPL 0.30\nMSFT 0.25\nSPY  0.25\nTLT  0.20",
        height=140,
        help="Weights are normalized to sum to 1.0 automatically.",
    )

    today = date.today()
    man_start, man_end = st.sidebar.date_input(
        "Date range",
        value=(today - timedelta(days=365 * 5), today),
        min_value=date(1990, 1, 1),
        max_value=today,
    )

    def _parse_weights(text: str) -> pd.Series | None:
        pairs = []
        for token in re.split(r"[,\n]+", text.strip()):
            parts = token.strip().split()
            if len(parts) == 2:
                try:
                    pairs.append((parts[0].upper(), float(parts[1])))
                except ValueError:
                    pass
        if not pairs:
            return None
        tks, wts = zip(*pairs)
        w = np.array(wts, dtype=float)
        return None if w.sum() <= 0 else pd.Series(w / w.sum(), index=list(tks))

    weights_parsed = _parse_weights(raw_manual)
    if weights_parsed is None:
        st.warning("Could not parse weights. Use format: TICKER weight (one per line).")
        st.stop()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_manual(ticker: str, start: date, end: date) -> pd.Series:
        df = data.get_prices(ticker, start, end)
        return df["adj_close"] if not df.empty else pd.Series(dtype=float, name=ticker)

    with st.spinner("Fetching prices..."):
        man_series = {t: _fetch_manual(t, man_start, man_end) for t in weights_parsed.index}

    failed_man = [t for t, s in man_series.items() if s.empty]
    if failed_man:
        st.warning(f"No data for: {', '.join(failed_man)} — skipping.")

    price_df = pd.DataFrame({t: s for t, s in man_series.items() if not s.empty}).dropna()
    if price_df.empty:
        st.error("No price data available for the entered tickers.")
        st.stop()

    weights = weights_parsed.reindex(price_df.columns).dropna()
    weights /= weights.sum()
    returns_df = price_df.pct_change().dropna()
    cov = pf.covariance_matrix(returns_df)
    method_used = ""

# ─── Portfolio composition ────────────────────────────────────────────────────

st.header("Portfolio composition")

if method_used:
    st.caption(f"Method: {_METHOD_NAMES.get(method_used, method_used)}")

Sigma    = cov.values
w_vec    = weights.reindex(cov.index).fillna(0.0).values
port_var = float(w_vec @ Sigma @ w_vec)
rc_pct   = pd.Series(
    w_vec * (Sigma @ w_vec) / port_var * 100,
    index=cov.index,
).reindex(weights.index).fillna(0.0)

sorted_idx  = weights.sort_values(ascending=True).index
wt_vals     = weights.reindex(sorted_idx).values * 100
rc_vals     = rc_pct.reindex(sorted_idx).values
yticks      = sorted_idx.tolist()
chart_h     = max(260, 46 * len(weights))

comp_bar = go.Figure(go.Bar(
    x=wt_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in wt_vals],
    textposition="outside",
    marker_color=PRIMARY,
))
comp_bar.update_layout(
    title="Capital allocation (%)", xaxis_title="Weight (%)",
    height=chart_h, margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(wt_vals) * 1.25]),
)
apply_chart_theme(comp_bar)

comp_rc = go.Figure(go.Bar(
    x=rc_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in rc_vals],
    textposition="outside",
    marker_color=NEGATIVE,
))
comp_rc.update_layout(
    title="Risk contribution (% of portfolio vol)", xaxis_title="Risk contribution (%)",
    height=chart_h, margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(rc_vals) * 1.25]),
)
apply_chart_theme(comp_rc)

col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    st.plotly_chart(comp_bar, use_container_width=True)
with col_comp2:
    st.plotly_chart(comp_rc, use_container_width=True)

if "risk_parity" in method_used:
    st.info(
        "For risk-parity portfolios, capital allocation and risk contribution are "
        "designed to be equal across assets by construction. Significant divergence "
        "between the two charts indicates a numerical issue in the solver."
    )

# ─── VaR / CVaR ──────────────────────────────────────────────────────────────

st.header("Value at Risk")

var_result = risk.portfolio_var(weights, returns_df)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Hist VaR 95%",   f"{var_result.hist_var_95 * 100:.2f}%")
m2.metric("Hist CVaR 95%",  f"{var_result.hist_cvar_95 * 100:.2f}%")
m3.metric("Hist VaR 99%",   f"{var_result.hist_var_99 * 100:.2f}%")
m4.metric("Hist CVaR 99%",  f"{var_result.hist_cvar_99 * 100:.2f}%")
m5.metric("Param VaR 95%",  f"{var_result.param_var_95 * 100:.2f}%")
m6.metric("Param VaR 99%",  f"{var_result.param_var_99 * 100:.2f}%")

port_rets_series = (
    returns_df.reindex(columns=weights.index).fillna(0.0) @ weights.values
)
hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=port_rets_series * 100,
    nbinsx=80,
    histnorm="probability density",
    name="Daily returns",
    opacity=0.72,
    marker_color=PRIMARY,
))
for label, val, color in [
    ("Hist VaR 95%",  var_result.hist_var_95,  BENCHMARK),
    ("Hist VaR 99%",  var_result.hist_var_99,  NEGATIVE),
    ("Param VaR 95%", var_result.param_var_95, PRIMARY),
]:
    hist_fig.add_vline(
        x=val * 100, line_color=color, line_dash="dash",
        annotation_text=f"{label}: {val * 100:.2f}%",
        annotation_position="top left",
    )
hist_fig.update_layout(
    xaxis_title="Daily return (%)",
    yaxis_title="Density",
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)
apply_chart_theme(hist_fig)
st.plotly_chart(hist_fig, use_container_width=True)

# ─── Monte Carlo ──────────────────────────────────────────────────────────────

st.header("Monte Carlo simulation  (5,000 paths, 1-year horizon)")

@st.cache_data(ttl=3600, show_spinner=False)
def _run_mc(w: pd.Series, rets: pd.DataFrame) -> np.ndarray:
    return risk.monte_carlo_paths(w, rets, n_paths=5000, horizon_days=252)


with st.spinner("Running 5,000 simulations..."):
    wealth = _run_mc(weights, returns_df)

days     = np.arange(wealth.shape[1])
terminal = wealth[:, -1]
p5       = np.percentile(wealth, 5,  axis=0)
p25      = np.percentile(wealth, 25, axis=0)
p50      = np.percentile(wealth, 50, axis=0)
p75      = np.percentile(wealth, 75, axis=0)
p95      = np.percentile(wealth, 95, axis=0)

worst5_idx = np.argsort(terminal)[:250]

mc_fig = go.Figure()

mc_fig.add_trace(go.Scatter(
    x=np.concatenate([days, days[::-1]]),
    y=np.concatenate([p95, p5[::-1]]),
    fill="toself", fillcolor=PRIMARY_10,
    line=dict(color="rgba(0,0,0,0)"),
    name="5th–95th percentile", showlegend=True,
))
mc_fig.add_trace(go.Scatter(
    x=np.concatenate([days, days[::-1]]),
    y=np.concatenate([p75, p25[::-1]]),
    fill="toself", fillcolor=PRIMARY_18,
    line=dict(color="rgba(0,0,0,0)"),
    name="25th–75th percentile", showlegend=True,
))

wx, wy = [], []
for i in worst5_idx[:100]:
    wx.extend(days.tolist() + [None])
    wy.extend(wealth[i].tolist() + [None])
mc_fig.add_trace(go.Scatter(
    x=wx, y=wy, mode="lines",
    name="Worst 5% paths",
    line=dict(color="rgba(229,86,78,0.18)", width=0.8),
))

mc_fig.add_trace(go.Scatter(
    x=days, y=p50, mode="lines",
    name="Median", line=dict(color=PRIMARY, width=2.2),
))
mc_fig.add_trace(go.Scatter(
    x=days, y=p5, mode="lines",
    name="5th percentile", line=dict(color=NEGATIVE, width=1.5, dash="dash"),
))

mc_fig.update_layout(
    xaxis_title="Trading day",
    yaxis_title="Wealth ($1 invested)",
    height=440,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    legend=dict(x=0.02, y=0.98),
)
apply_chart_theme(mc_fig)
st.plotly_chart(mc_fig, use_container_width=True)
st.caption(
    "Simulation assumes returns are independent and identically distributed. "
    "Actual equity returns exhibit volatility clustering and heavier tails; "
    "i.i.d. bootstrap paths will understate the frequency and severity of "
    "drawdown episodes."
)

# Summary metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Median terminal wealth",    f"${p50[-1]:.2f}")
mc2.metric("5th-pctl terminal wealth",  f"${p5[-1]:.2f}")
mc3.metric("95th-pctl terminal wealth", f"${p95[-1]:.2f}")
mc4.metric("Probability of loss",       f"{(terminal < 1.0).mean() * 100:.1f}%")

# Terminal wealth histogram
tw_fig = go.Figure()
tw_fig.add_trace(go.Histogram(
    x=terminal, nbinsx=60, histnorm="probability density",
    opacity=0.72, marker_color=PRIMARY, name="Terminal wealth",
))
tw_fig.add_vline(x=p5[-1], line_color=NEGATIVE, line_dash="dash",
                 annotation_text=f"5th pctl: ${p5[-1]:.2f}")
tw_fig.add_vline(x=1.0, line_color="gray", line_dash="dot",
                 annotation_text="Break even")
tw_fig.update_layout(
    title="Terminal wealth distribution (1-year)",
    xaxis_title="Terminal wealth ($1 invested)",
    height=280,
    margin=dict(l=10, r=10, t=40, b=10),
    showlegend=False,
)
apply_chart_theme(tw_fig)
st.plotly_chart(tw_fig, use_container_width=True)

# ─── Factor exposure ──────────────────────────────────────────────────────────

st.header("Factor exposure  (Fama-French 3-factor)")

@st.cache_data(ttl=86400, show_spinner=False)
def _load_ff3(start: date, end: date) -> pd.DataFrame | None:
    try:
        return risk.load_ff3_factors(start, end)
    except Exception:
        return None


ff3_start = price_df.index.min().date()
ff3_end   = price_df.index.max().date()

with st.spinner("Downloading Fama-French factors (cached 24 h)..."):
    ff3 = _load_ff3(ff3_start, ff3_end)

if ff3 is None:
    st.warning(
        "Could not download Fama-French data — check your internet connection. "
        "Factor exposure is unavailable."
    )
else:
    try:
        fr = risk.factor_exposure(weights, price_df, ff3)

        req_days  = (ff3_end - ff3_start).days
        act_days  = (fr.regression_end - fr.regression_start).days
        if act_days < req_days * 0.95:
            st.warning(
                f"Requested window: {ff3_start} → {ff3_end}. "
                f"FF3 data only available through {fr.regression_end}. "
                f"Regression uses the overlap below."
            )

        st.markdown(
            f"**Regression window:** {fr.regression_start} to {fr.regression_end}, "
            f"{fr.n_obs:,} observations"
        )

        factors = ["Mkt-RF", "SMB", "HML"]
        reg_data = {
            "Alpha (ann.)":         (f"{fr.alpha_annual * 100:.2f}%",  f"{fr.alpha_tstat:.2f}"),
            "β  Market (Mkt-RF)":   (f"{fr.betas['Mkt-RF']:.3f}",      f"{fr.tstats['Mkt-RF']:.2f}"),
            "β  Size (SMB)":        (f"{fr.betas['SMB']:.3f}",          f"{fr.tstats['SMB']:.2f}"),
            "β  Value (HML)":       (f"{fr.betas['HML']:.3f}",          f"{fr.tstats['HML']:.2f}"),
            "R²":                   (f"{fr.r_squared:.3f}",             "—"),
            "Residual vol (ann.)":  (f"{fr.residual_vol_annual * 100:.2f}%", "—"),
        }
        reg_df = pd.DataFrame.from_dict(
            reg_data, orient="index", columns=["Value", "t-stat"]
        )

        se = {f: abs(fr.betas[f] / fr.tstats[f]) if abs(fr.tstats[f]) > 0.01 else 0.0
              for f in factors}
        beta_colors = [
            PRIMARY if fr.betas[f] >= 0 else NEGATIVE for f in factors
        ]
        beta_fig = go.Figure(go.Bar(
            x=[fr.betas[f] for f in factors],
            y=["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"],
            orientation="h",
            error_x=dict(type="data", array=[1.96 * se[f] for f in factors], visible=True),
            marker_color=beta_colors,
        ))
        beta_fig.add_vline(x=0, line_color="gray", line_dash="dot")
        beta_fig.update_layout(
            title="Factor betas  (±1.96 SE)",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        apply_chart_theme(beta_fig)

        col_reg, col_beta = st.columns([1, 1])
        with col_reg:
            st.dataframe(reg_df, use_container_width=True)
            st.caption(
                "β_mkt > 1 indicates above-market systematic risk. "
                "β_smb > 0 indicates a tilt toward smaller-cap stocks; "
                "β_hml > 0 indicates a value tilt. "
                "Factor loadings with |t-stat| > 2 are statistically significant "
                "at the 5% level."
            )
        with col_beta:
            st.plotly_chart(beta_fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Factor regression failed: {exc}")

# ─── Stress tests ─────────────────────────────────────────────────────────────

st.header("Historical stress tests")

stress_results = risk.stress_test(weights, price_df)
covered   = [r for r in stress_results if r.covered]
uncovered = [r for r in stress_results if not r.covered]

if uncovered:
    st.caption(
        "Windows outside your date range (no data): "
        + ", ".join(r.window for r in uncovered)
    )

if not covered:
    st.info("None of the stress windows fall within the selected date range.")
else:
    stress_rows = []
    for r in covered:
        stress_rows.append({
            "Window":             r.window,
            "Portfolio return":   f"{r.port_return * 100:.1f}%",
            "Equal-weight":       f"{r.equal_return * 100:.1f}%",
            "Portfolio max DD":   f"{r.port_max_dd * 100:.1f}%",
            "EW max DD":          f"{r.equal_max_dd * 100:.1f}%",
        })
    st.dataframe(
        pd.DataFrame(stress_rows).set_index("Window"),
        use_container_width=True,
    )

    windows  = [r.window for r in covered]
    port_ret = [r.port_return * 100 for r in covered]
    eq_ret   = [r.equal_return * 100 for r in covered]

    stress_fig = go.Figure([
        go.Bar(name="Portfolio",    x=windows, y=port_ret, marker_color=PRIMARY),
        go.Bar(name="Equal weight", x=windows, y=eq_ret,   marker_color=NEUTRAL),
    ])
    stress_fig.add_hline(y=0, line_color="gray", line_dash="dot")
    stress_fig.update_layout(
        barmode="group",
        title="Cumulative return during stress windows",
        yaxis_title="Return (%)",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    apply_chart_theme(stress_fig)
    st.plotly_chart(stress_fig, use_container_width=True)
