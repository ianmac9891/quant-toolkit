"""Strategy Simulation — historical simulation with estimation/validation segmentation."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import analysis
from src import backtest as bt
from src import data
from src import strategies
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEUTRAL, PRIMARY_10,
    CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Portfolio & Risk", "Strategy Simulation",
    "Historical simulation of allocation strategies with transaction costs and "
    "strict no-lookahead discipline. Performance is segmented into estimation "
    "and validation periods to expose overfitting.",
)

STRATEGY_LABELS = {
    "buy_and_hold": "Static Allocation (equal weight)",
    "ma_crossover": "Moving-Average Crossover",
    "momentum":     "Cross-Sectional Momentum",
    "walk_forward": "Walk-Forward Optimization",
}

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with st.form("simulation_params"):
    st.markdown('<p class="qrt-kicker">Simulation Parameters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.8])
    with c1:
        tickers = ui.ticker_list_input("Investment Universe", "SPY\nTLT\nGLD", height=110)
        add_spy = st.checkbox("Overlay S&P 500 (SPY) on the equity curve", value=True)
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            start_date, end_date = ui.date_range_input(
                "Simulation Window", today - timedelta(days=365 * 10), today)
            default_split = start_date + timedelta(days=int((end_date - start_date).days * 0.7))
            split_date = st.date_input(
                "Estimation / Validation Boundary",
                value=default_split,
                min_value=start_date + timedelta(days=60),
                max_value=end_date - timedelta(days=60),
                help="Statistics are computed separately before (estimation) and "
                     "after (validation) this date.",
            )
            rf = ui.rf_rate_input(key="sim_rf")
        with cc2:
            rebalance_freq = st.selectbox(
                "Rebalancing Frequency", ["D", "W", "M", "Q"], index=2,
                format_func={"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly"}.get,
            )
            cost_bps = st.slider("Transaction Cost (bps, one-way)", 0, 50, 10, step=1)
            strategy_key = st.selectbox("Strategy", list(STRATEGY_LABELS),
                                        format_func=STRATEGY_LABELS.get)

    st.markdown('<p class="qrt-kicker" style="margin-top:0.6rem">Strategy Specification</p>',
                unsafe_allow_html=True)
    strategy_params: dict = {}
    s1, s2, s3 = st.columns(3)
    if strategy_key == "ma_crossover":
        with s1:
            strategy_params["fast"] = int(st.number_input("Fast Window (sessions)", 5, 100, 20, step=5))
        with s2:
            strategy_params["slow"] = int(st.number_input("Slow Window (sessions)", 20, 252, 60, step=5))
    elif strategy_key == "momentum":
        with s1:
            strategy_params["lookback_months"] = int(st.slider("Formation Period (months)", 3, 24, 12))
        with s2:
            strategy_params["skip_months"] = int(st.slider("Reversal Skip (months)", 0, 3, 1))
        with s3:
            strategy_params["top_k"] = int(st.slider("Positions Held (top N)", 1,
                                                     max(len(tickers), 1),
                                                     min(3, max(len(tickers), 1))))
    elif strategy_key == "walk_forward":
        with s1:
            strategy_params["lookback_months"] = int(st.slider("Estimation Lookback (months)", 12, 60, 36))
            strategy_params["method"] = st.selectbox(
                "Objective", ["max_sharpe", "min_variance", "risk_parity"],
                format_func={"max_sharpe": "Maximum Sharpe Ratio",
                             "min_variance": "Minimum Variance",
                             "risk_parity": "Risk Parity"}.get,
            )
        with s2:
            wc_pct = st.slider("Single-Position Limit (%)", 10, 100, 100, step=5)
            strategy_params["cov_estimator"] = st.selectbox(
                "Covariance Estimator", ["ledoit_wolf", "oas", "sample"],
                format_func={"ledoit_wolf": "Ledoit-Wolf", "oas": "OAS", "sample": "Sample"}.get,
            )
        with s3:
            strategy_params["mean_estimator"] = st.selectbox(
                "Expected-Return Estimator", ["james_stein", "sample"],
                format_func={"james_stein": "James-Stein", "sample": "Sample"}.get,
            )
        strategy_params["weight_cap"] = (
            wc_pct / 100 if strategy_params["method"] != "risk_parity" else 1.0
        )
        strategy_params["rf"] = rf
        strategy_params["min_obs"] = 60
    else:
        st.caption("The static allocation strategy has no additional parameters.")

    submitted = st.form_submit_button("Run Simulation", type="primary")

# ── Validation ────────────────────────────────────────────────────────────────

if len(tickers) < 2:
    ui.banner("warn", "Specify at least two instruments.")
    st.stop()

if strategy_key == "ma_crossover" and strategy_params.get("fast", 0) >= strategy_params.get("slow", 1):
    ui.banner("error", "The fast window must be shorter than the slow window.")
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

spy_series: pd.Series | None = None
if add_spy:
    spy_raw = _fetch("SPY", start_date, end_date)
    if not spy_raw.empty:
        spy_aligned = spy_raw.reindex(price_df.index).ffill().bfill()
        spy_series = spy_aligned.dropna() if not spy_aligned.dropna().empty else None

# ── Run ───────────────────────────────────────────────────────────────────────

def _make_strategy(name: str, params: dict):
    if name == "buy_and_hold":
        return strategies.buy_and_hold()
    if name == "ma_crossover":
        return strategies.ma_crossover(params["fast"], params["slow"])
    if name == "momentum":
        return strategies.cross_sectional_momentum(
            params["lookback_months"], params["skip_months"], params["top_k"])
    return strategies.walk_forward_optimizer(
        lookback_months=params["lookback_months"],
        method=params["method"],
        rf=params["rf"],
        weight_cap=params["weight_cap"],
        cov_estimator=params["cov_estimator"],
        mean_estimator=params["mean_estimator"],
        min_obs=params["min_obs"],
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _run(price_data: pd.DataFrame, strategy_name: str, params: dict,
         rebal_freq: str, tx_cost_bps: float) -> bt.BacktestResult:
    return bt.run_backtest(price_data, _make_strategy(strategy_name, params),
                           rebalance_freq=rebal_freq, cost_bps=tx_cost_bps)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_ew(price_data: pd.DataFrame, rebal_freq: str) -> bt.BacktestResult:
    return bt.run_backtest(price_data, strategies.buy_and_hold(),
                           rebalance_freq=rebal_freq, cost_bps=0.0)


spinner_msg = (
    "Running walk-forward simulation (re-optimizing at each rebalance; "
    "approximately 30 seconds on first run)..."
    if strategy_key == "walk_forward"
    else "Running simulation..."
)
with st.spinner(spinner_msg):
    result    = _run(price_df, strategy_key, strategy_params, rebalance_freq, float(cost_bps))
    ew_result = _run_ew(price_df, rebalance_freq)

# ── Segmentation ──────────────────────────────────────────────────────────────

split_ts = pd.Timestamp(split_date)

first_active = (
    result.trade_log["date"].iloc[0]
    if not result.trade_log.empty
    else result.equity.index[0]
)

is_equity  = result.equity.loc[first_active:split_ts]
oos_equity = result.equity.loc[split_ts:]

is_trades  = result.trade_log[result.trade_log["date"] <= split_ts] if not result.trade_log.empty else result.trade_log
oos_trades = result.trade_log[result.trade_log["date"] >  split_ts] if not result.trade_log.empty else result.trade_log

ew_is_equity  = ew_result.equity.loc[first_active:split_ts]
ew_oos_equity = ew_result.equity.loc[split_ts:]
ew_trades     = ew_result.trade_log
ew_is_trades  = ew_trades[ew_trades["date"] <= split_ts] if not ew_trades.empty else ew_trades
ew_oos_trades = ew_trades[ew_trades["date"] >  split_ts] if not ew_trades.empty else ew_trades


def _norm100(s: pd.Series) -> pd.Series:
    return s / s.iloc[0] * 100


def _ts_ms(ts) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


# ── Equity curve ──────────────────────────────────────────────────────────────

with ui.panel("Equity Curve (indexed to 100, log scale)"):
    eq_fig = go.Figure()

    if first_active > result.equity.index[0]:
        eq_fig.add_vrect(
            x0=_ts_ms(result.equity.index[0]), x1=_ts_ms(first_active),
            fillcolor=NEUTRAL, opacity=0.15, line_width=0,
            annotation_text="Warm-up", annotation_position="top left",
        )
    eq_fig.add_vline(
        x=_ts_ms(split_ts), line_dash="dot", line_color=NEUTRAL, line_width=1.5,
        annotation_text="Estimation | Validation", annotation_position="top right",
    )
    eq_fig.add_trace(go.Scatter(
        x=result.equity.index, y=_norm100(result.equity),
        mode="lines", name=STRATEGY_LABELS[strategy_key],
        line=dict(color=PRIMARY, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
    ))
    eq_fig.add_trace(go.Scatter(
        x=ew_result.equity.index, y=_norm100(ew_result.equity),
        mode="lines", name="Equal weight (no costs)",
        line=dict(color=POSITIVE, width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
    ))
    if spy_series is not None:
        eq_fig.add_trace(go.Scatter(
            x=spy_series.index, y=_norm100(spy_series),
            mode="lines", name="S&P 500 (SPY)",
            line=dict(color=BENCHMARK, width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
        ))
    eq_fig.update_layout(
        yaxis_type="log", yaxis_title="Portfolio value (indexed)",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(eq_fig)
    st.plotly_chart(eq_fig, width="stretch", config=CHART_CONFIG)
    st.caption(
        "The shaded region marks the strategy warm-up period, which is excluded "
        "from estimation-period statistics. The dotted vertical line is the "
        "estimation/validation boundary. The equal-weight reference is simulated "
        "without transaction costs."
    )

# ── Drawdown ──────────────────────────────────────────────────────────────────

def _dd_series(equity: pd.Series) -> pd.Series:
    rets = equity.pct_change().dropna()
    return analysis.drawdown(rets).series * 100 if not rets.empty else pd.Series(dtype=float)


with ui.panel("Drawdown"):
    dd_fig = go.Figure()
    if first_active > result.equity.index[0]:
        dd_fig.add_vrect(
            x0=_ts_ms(result.equity.index[0]), x1=_ts_ms(first_active),
            fillcolor=NEUTRAL, opacity=0.15, line_width=0,
        )
    dd_fig.add_vline(x=_ts_ms(split_ts), line_dash="dot", line_color=NEUTRAL, line_width=1.5)

    strat_dd = _dd_series(result.equity.loc[first_active:])
    ew_dd    = _dd_series(ew_result.equity.loc[first_active:])

    dd_fig.add_trace(go.Scatter(
        x=strat_dd.index, y=strat_dd, mode="lines",
        name=STRATEGY_LABELS[strategy_key],
        line=dict(color=PRIMARY, width=2), fill="tozeroy", fillcolor=PRIMARY_10,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    dd_fig.add_trace(go.Scatter(
        x=ew_dd.index, y=ew_dd, mode="lines", name="Equal weight",
        line=dict(color=POSITIVE, width=1.5, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    dd_fig.update_layout(
        yaxis_title="Drawdown (%)", height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.05),
    )
    apply_chart_theme(dd_fig)
    st.plotly_chart(dd_fig, width="stretch", config=CHART_CONFIG)

# ── Performance segmentation ──────────────────────────────────────────────────

if len(is_equity) < 5:
    ui.banner(
        "warn",
        "The estimation period contains very few observations after warm-up. "
        "Move the boundary later for reliable estimation-period statistics.",
    )


def _fmt_stats(stats: dict) -> dict:
    def _fmt(k: str, v: float) -> str:
        if not np.isfinite(v):
            return "—"
        if k in ("Ann. return", "Ann. vol", "Max drawdown"):
            return f"{v * 100:.1f}%"
        if k == "Avg daily turnover":
            return f"{v * 100:.2f}%"
        return f"{v:.2f}"
    return {k: _fmt(k, v) for k, v in stats.items()}


is_stats     = bt.perf_stats(is_equity,     is_trades,     rf=rf)
oos_stats    = bt.perf_stats(oos_equity,    oos_trades,    rf=rf)
ew_is_stats  = bt.perf_stats(ew_is_equity,  ew_is_trades,  rf=rf)
ew_oos_stats = bt.perf_stats(ew_oos_equity, ew_oos_trades, rf=rf)

with ui.panel("Performance — Estimation vs Validation"):
    perf_table = pd.DataFrame({
        "Estimation — Strategy":    _fmt_stats(is_stats),
        "Validation — Strategy":    _fmt_stats(oos_stats),
        "Estimation — Equal Weight": _fmt_stats(ew_is_stats),
        "Validation — Equal Weight": _fmt_stats(ew_oos_stats),
    })
    st.dataframe(perf_table, width="stretch")

    is_sharpe  = is_stats.get("Sharpe",  float("nan"))
    oos_sharpe = oos_stats.get("Sharpe", float("nan"))
    if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe):
        if is_sharpe > 0 and oos_sharpe < 0:
            ui.banner(
                "error",
                f"The validation Sharpe ratio ({oos_sharpe:.2f}) is negative while "
                f"the estimation Sharpe ({is_sharpe:.2f}) was positive — a strong "
                "indication of overfitting or a structural regime change.",
            )
        elif is_sharpe > 0.5 and oos_sharpe < is_sharpe * 0.5:
            ui.banner(
                "warn",
                f"The validation Sharpe ratio ({oos_sharpe:.2f}) is less than half "
                f"the estimation Sharpe ({is_sharpe:.2f}). The parameters may not "
                "generalize beyond the fitted period.",
            )

# ── Calendar-year returns ─────────────────────────────────────────────────────

def _yearly_returns(equity: pd.Series) -> pd.Series:
    rets = equity.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    return (1 + rets).groupby(rets.index.year).prod() - 1


with ui.panel("Calendar-Year Returns"):
    yr_cols = {
        STRATEGY_LABELS[strategy_key]: _yearly_returns(result.equity.loc[first_active:]),
        "Equal Weight": _yearly_returns(ew_result.equity.loc[first_active:]),
    }
    if spy_series is not None:
        yr_cols["S&P 500 (SPY)"] = _yearly_returns(spy_series.loc[first_active:])

    yr_df = pd.DataFrame(yr_cols)
    if yr_df.empty:
        ui.banner("info", "Not enough history for calendar-year statistics.")
    else:
        yr_df.index.name = "Year"
        disp_yr = (yr_df * 100).round(1)
        st.dataframe(
            disp_yr,
            column_config={
                c: st.column_config.NumberColumn(c, format="%.1f%%") for c in disp_yr.columns
            },
            width="stretch",
        )
        st.caption(
            "Partial first and last years cover only the simulated portion of the "
            "calendar year. The warm-up period is excluded."
        )

# ── Ledger and weights ────────────────────────────────────────────────────────

n_trades = len(result.trade_log)
with st.expander(f"Rebalance Ledger ({n_trades} rebalance{'s' if n_trades != 1 else ''})"):
    if result.trade_log.empty:
        ui.banner("info", "The strategy held cash for the entire period; no rebalances recorded.")
    else:
        display_log = result.trade_log.copy()
        display_log["turnover"] = display_log["turnover"].map("{:.1%}".format)
        display_log["cost_pct"] = display_log["cost_pct"].map("{:.3%}".format)
        st.dataframe(display_log, width="stretch")

with st.expander("Target Weight History"):
    wh = result.weights_history
    if wh.empty:
        ui.banner("info", "No rebalances recorded.")
    else:
        wh_pct = (wh * 100).round(1)
        wh_fig = go.Figure(go.Heatmap(
            z=wh_pct.values,
            x=wh_pct.columns.tolist(),
            y=[str(d.date()) if hasattr(d, "date") else str(d) for d in wh_pct.index],
            colorscale="Blues", zmin=0, zmax=100,
            text=wh_pct.values, texttemplate="%{text:.0f}%",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        ))
        wh_fig.update_layout(
            title="Target weights at each rebalance (%)",
            height=max(300, 22 * len(wh_pct)),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        apply_chart_theme(wh_fig)
        st.plotly_chart(wh_fig, width="stretch", config=CHART_CONFIG)

ui.footer_disclaimer()
