"""Risk Analytics — VaR/CVaR, forward wealth simulation, factor exposure, stress replay."""

from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from src import analysis, portfolio as pf, risk
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE,
    PRIMARY_10, PRIMARY_18, CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Portfolio & Risk", "Risk Analytics",
    "Value at Risk and expected shortfall, Monte Carlo forward wealth "
    "simulation, Fama-French three-factor exposure, and replay of historical "
    "stress windows for a stated portfolio.",
)

_METHOD_NAMES = {
    "max_sharpe":              "Maximum Sharpe Ratio",
    "min_variance":            "Minimum Variance",
    "risk_parity":             "Risk Parity",
    "max_sharpe_resampled":    "Maximum Sharpe Ratio (Michaud resampled)",
    "min_variance_resampled":  "Minimum Variance (Michaud resampled)",
    "risk_parity_resampled":   "Risk Parity (Michaud resampled)",
}

# ── Portfolio source ──────────────────────────────────────────────────────────

with ui.panel("Portfolio Definition"):
    input_mode = st.radio(
        "Source",
        ["Staged from Portfolio Construction", "Manual entry"],
        horizontal=True,
    )

    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        portfolio_value = st.number_input(
            "Portfolio Value ($)", min_value=1_000.0, value=1_000_000.0,
            step=50_000.0, format="%.0f",
            help="Notional used to express risk figures in dollar terms.",
        )
    with pcol2:
        var_horizon = st.selectbox(
            "VaR Horizon", options=[1, 5, 10, 21],
            format_func=lambda d: {1: "1 day", 5: "5 days (1 week)",
                                   10: "10 days (regulatory)", 21: "21 days (1 month)"}[d],
            help="Multi-day figures use square-root-of-time scaling, which assumes "
                 "independent returns and understates risk under volatility clustering.",
        )

    if input_mode == "Staged from Portfolio Construction":
        if "portfolio_weights" not in st.session_state:
            ui.banner(
                "info",
                "No portfolio staged in this session. Construct one in "
                "<b>Portfolio Construction</b> first, or switch to manual entry.",
            )
            st.stop()
        weights     = st.session_state["portfolio_weights"]
        price_df    = st.session_state["portfolio_prices"]
        returns_df  = st.session_state["portfolio_returns"]
        method_used = st.session_state.get("portfolio_method", "")
        _cov_stored = st.session_state.get("portfolio_cov")
        cov         = _cov_stored if _cov_stored is not None else pf.covariance_matrix(returns_df)
        if method_used:
            st.caption(f"Staged allocation: {_METHOD_NAMES.get(method_used, method_used)}")
    else:
        mcol1, mcol2 = st.columns([1.2, 1])
        with mcol1:
            raw_manual = st.text_area(
                "Holdings (symbol and weight, one per line)",
                "AAPL 0.30\nMSFT 0.25\nSPY  0.25\nTLT  0.20",
                height=140,
                help="Weights are normalized to sum to one.",
            )
        with mcol2:
            today = date.today()
            man_start, man_end = ui.date_range_input(
                "Estimation Window", today - timedelta(days=365 * 5), today,
                key="risk_manual_range",
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
            ui.banner("warn", "Could not parse holdings. Use the format: "
                              "<span class='mono'>SYMBOL weight</span>, one per line.")
            st.stop()

        with st.spinner("Retrieving price histories..."):
            frames = ui.fetch_universe(tuple(weights_parsed.index), man_start, man_end)

        man_series = {
            t: df["adj_close"].rename(t)
            for t, df in frames.items()
            if not df.empty and "adj_close" in df.columns
        }
        failed_man = sorted(set(weights_parsed.index) - set(man_series))
        if failed_man:
            ui.banner("warn", f"No data for: <span class='mono'>{', '.join(failed_man)}</span> — excluded.")

        price_df = pd.DataFrame(man_series).dropna()
        if price_df.empty:
            ui.data_unavailable()
            st.stop()

        weights = weights_parsed.reindex(price_df.columns).dropna()
        weights /= weights.sum()
        returns_df = price_df.pct_change().dropna()
        cov = pf.covariance_matrix(returns_df)
        method_used = ""

ui.data_asof_caption(price_df.index.max())

# ── Composition ───────────────────────────────────────────────────────────────

Sigma    = cov.values
w_vec    = weights.reindex(cov.index).fillna(0.0).values
port_var = float(w_vec @ Sigma @ w_vec)
rc_pct   = pd.Series(
    w_vec * (Sigma @ w_vec) / port_var * 100, index=cov.index,
).reindex(weights.index).fillna(0.0)

sorted_idx = weights.sort_values(ascending=True).index
wt_vals    = weights.reindex(sorted_idx).values * 100
rc_vals    = rc_pct.reindex(sorted_idx).values
yticks     = sorted_idx.tolist()
chart_h    = max(260, 46 * len(weights))

comp_bar = go.Figure(go.Bar(
    x=wt_vals, y=yticks, orientation="h",
    text=[f"{v:.1f}%" for v in wt_vals], textposition="outside",
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
    text=[f"{v:.1f}%" for v in rc_vals], textposition="outside",
    marker_color=NEGATIVE,
))
comp_rc.update_layout(
    title="Risk contribution (% of portfolio variance)", xaxis_title="Risk contribution (%)",
    height=chart_h, margin=dict(l=10, r=70, t=40, b=10),
    xaxis=dict(range=[0, max(rc_vals) * 1.25]),
)
apply_chart_theme(comp_rc)

col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    with ui.panel("Composition"):
        st.plotly_chart(comp_bar, width="stretch", config=CHART_CONFIG)
with col_comp2:
    with ui.panel("Risk Decomposition"):
        st.plotly_chart(comp_rc, width="stretch", config=CHART_CONFIG)

if "risk_parity" in method_used:
    ui.banner(
        "info",
        "Risk-parity allocations equalize risk contribution by construction; "
        "material divergence between the two charts would indicate a solver issue.",
    )

# ── Historical performance ────────────────────────────────────────────────────

hist_rets = (returns_df.reindex(columns=weights.index).fillna(0.0) @ weights.values)
hist_rets = pd.Series(hist_rets, index=returns_df.index).dropna()

if len(hist_rets) > 60:
    dd_res = analysis.drawdown(hist_rets)
    wealth_hist = analysis.cumulative_returns(hist_rets)

    ui.kpi_row([
        {"label": "Annualized Return", "value": f"{analysis.annualized_return(hist_rets) * 100:.2f}%"},
        {"label": "Annualized Volatility", "value": f"{analysis.annualized_volatility(hist_rets) * 100:.2f}%"},
        {"label": "Maximum Drawdown", "value": f"{dd_res.max_drawdown * 100:.1f}%",
         "delta_kind": "neg"},
        {"label": "Worst Day", "value": f"{hist_rets.min() * 100:.2f}%", "delta_kind": "neg"},
        {"label": "Best Day", "value": f"{hist_rets.max() * 100:+.2f}%", "delta_kind": "pos"},
    ])

    with ui.panel("Historical Portfolio Performance (current weights, daily rebalanced)"):
        hp_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.05,
            subplot_titles=("Growth of $1", "Drawdown"),
        )
        hp_fig.add_trace(go.Scatter(
            x=wealth_hist.index, y=wealth_hist.values, name="Portfolio",
            line=dict(color=PRIMARY, width=1.5),
        ), row=1, col=1)
        hp_fig.add_trace(go.Scatter(
            x=dd_res.series.index, y=dd_res.series.values * 100,
            name="Drawdown %", fill="tozeroy", fillcolor=PRIMARY_10,
            line=dict(color=PRIMARY, width=0.5),
        ), row=2, col=1)
        hp_fig.update_layout(
            height=400, margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified", showlegend=False,
        )
        hp_fig.update_yaxes(title_text="$", row=1, col=1)
        hp_fig.update_yaxes(title_text="%", row=2, col=1)
        apply_chart_theme(hp_fig)
        st.plotly_chart(hp_fig, width="stretch", config=CHART_CONFIG)
        st.caption(
            f"Maximum drawdown {dd_res.max_drawdown * 100:.1f}%: peak "
            f"{dd_res.peak_date.date()}, trough {dd_res.trough_date.date()}. "
            "Applies today's weights to the full return history with daily "
            "rebalancing and no transaction costs."
        )

# ── Value at Risk ─────────────────────────────────────────────────────────────

var_result = risk.portfolio_var(weights, returns_df)
scale = np.sqrt(var_horizon)
h_label = f"{var_horizon}-day" if var_horizon > 1 else "1-day"

ui.section(f"Value at Risk — {h_label} horizon, ${portfolio_value:,.0f} notional")

ui.kpi_row([
    {"label": f"Hist VaR 95% ({h_label})",
     "value": f"{var_result.hist_var_95 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.hist_var_95) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
    {"label": f"Hist CVaR 95% ({h_label})",
     "value": f"{var_result.hist_cvar_95 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.hist_cvar_95) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
    {"label": f"Hist VaR 99% ({h_label})",
     "value": f"{var_result.hist_var_99 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.hist_var_99) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
    {"label": f"Hist CVaR 99% ({h_label})",
     "value": f"{var_result.hist_cvar_99 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.hist_cvar_99) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
    {"label": f"Param VaR 95% ({h_label})",
     "value": f"{var_result.param_var_95 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.param_var_95) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
    {"label": f"Param VaR 99% ({h_label})",
     "value": f"{var_result.param_var_99 * scale * 100:.2f}%",
     "delta": f"${abs(var_result.param_var_99) * scale * portfolio_value:,.0f}",
     "delta_kind": "neg"},
])
if var_horizon > 1:
    st.caption(
        "Multi-day figures apply square-root-of-time scaling to the one-day "
        "estimates. This assumes serially independent returns; under volatility "
        "clustering it understates multi-day tail risk."
    )

with ui.panel("Daily Return Distribution and VaR Thresholds (1-day)"):
    port_rets_series = (
        returns_df.reindex(columns=weights.index).fillna(0.0) @ weights.values
    )
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=port_rets_series * 100, nbinsx=80, histnorm="probability density",
        name="Daily returns", opacity=0.72, marker_color=PRIMARY,
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
        xaxis_title="Daily return (%)", yaxis_title="Density",
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    apply_chart_theme(hist_fig)
    st.plotly_chart(hist_fig, width="stretch", config=CHART_CONFIG)

# ── Monte Carlo ───────────────────────────────────────────────────────────────

ui.section("Forward Wealth Simulation")

with ui.panel("Simulation Parameters"):
    sc1, sc2 = st.columns(2)
    with sc1:
        n_paths = st.select_slider("Simulation Paths", options=[1000, 2500, 5000, 10000],
                                   value=5000)
    with sc2:
        mc_horizon = st.select_slider(
            "Horizon (sessions)", options=[63, 126, 252, 504], value=252,
            help="252 sessions is approximately one calendar year.",
        )


@st.cache_data(ttl=3600, show_spinner=False)
def _run_mc(w: pd.Series, rets: pd.DataFrame, n_paths: int, horizon: int) -> np.ndarray:
    return risk.monte_carlo_paths(w, rets, n_paths=n_paths, horizon_days=horizon)


with st.spinner(f"Simulating {n_paths:,} paths..."):
    wealth = _run_mc(weights, returns_df, n_paths, mc_horizon)

days     = np.arange(wealth.shape[1])
terminal = wealth[:, -1]
p5  = np.percentile(wealth, 5,  axis=0)
p25 = np.percentile(wealth, 25, axis=0)
p50 = np.percentile(wealth, 50, axis=0)
p75 = np.percentile(wealth, 75, axis=0)
p95 = np.percentile(wealth, 95, axis=0)

worst5_idx = np.argsort(terminal)[: max(int(n_paths * 0.05), 10)]

with ui.panel(f"Wealth Paths — {n_paths:,} simulations, {mc_horizon} sessions"):
    mc_fig = go.Figure()
    mc_fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor=PRIMARY_10,
        line=dict(color="rgba(0,0,0,0)"),
        name="5th-95th percentile", showlegend=True,
    ))
    mc_fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor=PRIMARY_18,
        line=dict(color="rgba(0,0,0,0)"),
        name="25th-75th percentile", showlegend=True,
    ))

    wx, wy = [], []
    for i in worst5_idx[:100]:
        wx.extend(days.tolist() + [None])
        wy.extend(wealth[i].tolist() + [None])
    mc_fig.add_trace(go.Scatter(
        x=wx, y=wy, mode="lines", name="Worst 5% of paths",
        line=dict(color="rgba(229,86,78,0.18)", width=0.8),
    ))
    mc_fig.add_trace(go.Scatter(
        x=days, y=p50, mode="lines", name="Median",
        line=dict(color=PRIMARY, width=2.2),
    ))
    mc_fig.add_trace(go.Scatter(
        x=days, y=p5, mode="lines", name="5th percentile",
        line=dict(color=NEGATIVE, width=1.5, dash="dash"),
    ))
    mc_fig.update_layout(
        xaxis_title="Session", yaxis_title="Wealth ($1 invested)",
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(mc_fig)
    st.plotly_chart(mc_fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Paths are an i.i.d. bootstrap of historical daily portfolio returns. "
        "Actual returns exhibit volatility clustering and heavier tails; the "
        "simulation understates the frequency and depth of drawdown episodes."
    )

ui.kpi_row([
    {"label": "Median Terminal Wealth", "value": f"${p50[-1] * portfolio_value:,.0f}"},
    {"label": "5th Percentile", "value": f"${p5[-1] * portfolio_value:,.0f}"},
    {"label": "95th Percentile", "value": f"${p95[-1] * portfolio_value:,.0f}"},
    {"label": "Probability of Loss", "value": f"{(terminal < 1.0).mean() * 100:.1f}%"},
])

with ui.panel("Terminal Wealth Distribution"):
    tw_fig = go.Figure()
    tw_fig.add_trace(go.Histogram(
        x=terminal, nbinsx=60, histnorm="probability density",
        opacity=0.72, marker_color=PRIMARY, name="Terminal wealth",
    ))
    tw_fig.add_vline(x=p5[-1], line_color=NEGATIVE, line_dash="dash",
                     annotation_text=f"5th pctl: {p5[-1]:.2f}x")
    tw_fig.add_vline(x=1.0, line_color=REFLINE, line_dash="dot",
                     annotation_text="Break even")
    tw_fig.update_layout(
        xaxis_title="Terminal wealth (multiple of initial)",
        height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
    )
    apply_chart_theme(tw_fig)
    st.plotly_chart(tw_fig, width="stretch", config=CHART_CONFIG)

# ── Factor exposure ───────────────────────────────────────────────────────────

ui.section("Factor Exposure — Fama-French Three-Factor Model")


@st.cache_data(ttl=86400, show_spinner=False)
def _load_ff3(start: date, end: date) -> pd.DataFrame | None:
    try:
        return risk.load_ff3_factors(start, end)
    except Exception:
        return None


ff3_start = price_df.index.min().date()
ff3_end   = price_df.index.max().date()

with st.spinner("Retrieving Fama-French factor returns (cached 24 hours)..."):
    ff3 = _load_ff3(ff3_start, ff3_end)

if ff3 is None:
    ui.banner("warn", "Fama-French factor data could not be retrieved — "
                      "factor exposure is unavailable.")
else:
    try:
        fr = risk.factor_exposure(weights, price_df, ff3)

        req_days = (ff3_end - ff3_start).days
        act_days = (fr.regression_end - fr.regression_start).days
        if act_days < req_days * 0.95:
            ui.banner(
                "warn",
                f"Requested window {ff3_start} to {ff3_end}; published factor data "
                f"ends {fr.regression_end}. The regression uses the overlap.",
            )

        with ui.panel(f"Regression — {fr.regression_start} to {fr.regression_end}, "
                      f"{fr.n_obs:,} observations"):
            factors = ["Mkt-RF", "SMB", "HML"]
            reg_data = {
                "Alpha (annualized)":   (f"{fr.alpha_annual * 100:.2f}%", f"{fr.alpha_tstat:.2f}"),
                "Beta: Market (Mkt-RF)": (f"{fr.betas['Mkt-RF']:.3f}", f"{fr.tstats['Mkt-RF']:.2f}"),
                "Beta: Size (SMB)":      (f"{fr.betas['SMB']:.3f}",    f"{fr.tstats['SMB']:.2f}"),
                "Beta: Value (HML)":     (f"{fr.betas['HML']:.3f}",    f"{fr.tstats['HML']:.2f}"),
                "R²":                    (f"{fr.r_squared:.3f}", "—"),
                "Residual Vol (annualized)": (f"{fr.residual_vol_annual * 100:.2f}%", "—"),
            }
            reg_df = pd.DataFrame.from_dict(reg_data, orient="index", columns=["Estimate", "t-stat"])

            se = {f: abs(fr.betas[f] / fr.tstats[f]) if abs(fr.tstats[f]) > 0.01 else 0.0
                  for f in factors}
            beta_colors = [PRIMARY if fr.betas[f] >= 0 else NEGATIVE for f in factors]
            beta_fig = go.Figure(go.Bar(
                x=[fr.betas[f] for f in factors],
                y=["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"],
                orientation="h",
                error_x=dict(type="data", array=[1.96 * se[f] for f in factors], visible=True),
                marker_color=beta_colors,
            ))
            beta_fig.add_vline(x=0, line_color=REFLINE, line_dash="dot")
            beta_fig.update_layout(
                title="Factor loadings (±1.96 SE)", height=260,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            apply_chart_theme(beta_fig)

            col_reg, col_beta = st.columns([1, 1])
            with col_reg:
                st.dataframe(reg_df, width="stretch")
                st.caption(
                    "A market beta above one indicates above-market systematic risk; "
                    "a positive SMB loading indicates a small-cap tilt and a positive "
                    "HML loading a value tilt. Loadings with |t| > 2 are significant "
                    "at the 5% level."
                )
            with col_beta:
                st.plotly_chart(beta_fig, width="stretch", config=CHART_CONFIG)

    except Exception as exc:
        ui.banner("error", f"Factor regression failed: {exc}")

# ── Stress scenarios ──────────────────────────────────────────────────────────

ui.section("Historical Stress Scenarios")

stress_results = risk.stress_test(weights, price_df)
covered   = [s for s in stress_results if s.covered]
uncovered = [s for s in stress_results if not s.covered]

if uncovered:
    st.caption("Scenarios outside the available price history: "
               + ", ".join(s.window for s in uncovered))

if not covered:
    ui.banner("info", "None of the predefined stress windows fall within the "
                      "portfolio's price history.")
else:
    with ui.panel("Scenario Replay"):
        stress_rows = []
        for s in covered:
            stress_rows.append({
                "Scenario":           s.window,
                "Portfolio Return":   f"{s.port_return * 100:.1f}%",
                "Equal-Weight":       f"{s.equal_return * 100:.1f}%",
                "Portfolio Max DD":   f"{s.port_max_dd * 100:.1f}%",
                "Equal-Weight Max DD": f"{s.equal_max_dd * 100:.1f}%",
                "Dollar P&L":         f"${s.port_return * portfolio_value:,.0f}",
            })
        st.dataframe(pd.DataFrame(stress_rows).set_index("Scenario"),
                     width="stretch")

        windows  = [s.window for s in covered]
        port_ret = [s.port_return * 100 for s in covered]
        eq_ret   = [s.equal_return * 100 for s in covered]

        stress_fig = go.Figure([
            go.Bar(name="Portfolio",    x=windows, y=port_ret, marker_color=PRIMARY),
            go.Bar(name="Equal weight", x=windows, y=eq_ret,   marker_color=NEUTRAL),
        ])
        stress_fig.add_hline(y=0, line_color=REFLINE, line_dash="dot")
        stress_fig.update_layout(
            barmode="group", title="Cumulative return through each stress window",
            yaxis_title="Return (%)", height=340, margin=dict(l=10, r=10, t=40, b=10),
        )
        apply_chart_theme(stress_fig)
        st.plotly_chart(stress_fig, width="stretch", config=CHART_CONFIG)

ui.footer_disclaimer()
