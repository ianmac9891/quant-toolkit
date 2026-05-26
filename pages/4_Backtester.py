"""Backtesting page: IS/OOS performance evaluation."""

from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analysis
from src import backtest as bt
from src import data
from src import strategies
from src.theme import PRIMARY, BENCHMARK, POSITIVE, NEUTRAL, PRIMARY_10, apply_chart_theme

st.set_page_config(page_title="Backtester", layout="wide")

STRATEGY_LABELS = {
    "buy_and_hold": "Buy and Hold (equal weight)",
    "ma_crossover": "MA Crossover",
    "momentum":     "Cross-sectional Momentum",
    "walk_forward": "Walk-forward Optimizer",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Inputs")

raw_tickers = st.sidebar.text_area(
    "Tickers",
    "SPY\nTLT\nGLD",
    height=100,
    help="One per line or comma-separated.",
)
tickers = sorted(set(t for t in re.split(r"[\s,]+", raw_tickers.strip().upper()) if t))

today = date.today()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(today - timedelta(days=365 * 10), today),
    min_value=date(1990, 1, 1),
    max_value=today,
)

default_split = start_date + timedelta(days=int((end_date - start_date).days * 0.7))
split_date = st.sidebar.date_input(
    "IS / OOS split",
    value=default_split,
    min_value=start_date + timedelta(days=60),
    max_value=end_date - timedelta(days=60),
    help="Performance metrics are computed separately for the in-sample (before this date) and out-of-sample (after this date) periods.",
)

rf_pct = st.sidebar.number_input(
    "Risk-free rate (% annual)", min_value=0.0, max_value=20.0, value=4.5, step=0.25,
)
rf = rf_pct / 100

rebalance_freq = st.sidebar.selectbox(
    "Rebalance frequency",
    ["D", "W", "M", "Q"],
    index=2,
    format_func={"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly"}.get,
)

cost_bps = st.sidebar.slider(
    "Transaction cost (bps, one-way)", min_value=0, max_value=50, value=10, step=1,
)

add_spy = st.sidebar.checkbox("Add SPY benchmark to equity chart", value=True)

# ── Strategy selector ─────────────────────────────────────────────────────────

strategy_key = st.sidebar.selectbox(
    "Strategy",
    list(STRATEGY_LABELS),
    format_func=STRATEGY_LABELS.get,
)

strategy_params: dict = {}
with st.sidebar.expander("Strategy parameters", expanded=True):
    if strategy_key == "ma_crossover":
        strategy_params["fast"] = int(st.number_input("Fast window (days)", 5, 100, 20, step=5))
        strategy_params["slow"] = int(st.number_input("Slow window (days)", 20, 252, 60, step=5))
        if strategy_params["fast"] >= strategy_params["slow"]:
            st.warning("Fast window must be shorter than slow window.")
    elif strategy_key == "momentum":
        strategy_params["lookback_months"] = int(st.slider("Lookback (months)", 3, 24, 12))
        strategy_params["skip_months"]     = int(st.slider("Skip months (reversal buffer)", 0, 3, 1))
        strategy_params["top_k"]           = int(st.slider("Top N assets to hold", 1, max(len(tickers), 1), min(3, max(len(tickers), 1))))
    elif strategy_key == "walk_forward":
        strategy_params["lookback_months"] = int(st.slider("Lookback window (months)", 12, 60, 36))
        strategy_params["method"] = st.selectbox(
            "Optimizer",
            ["max_sharpe", "min_variance", "risk_parity"],
            format_func={"max_sharpe": "Max Sharpe", "min_variance": "Min Variance", "risk_parity": "Risk Parity"}.get,
        )
        wc_pct = st.slider("Max weight per asset (%)", 10, 100, 100, step=5,
                           disabled=(strategy_params["method"] == "risk_parity"))
        strategy_params["weight_cap"]      = wc_pct / 100 if strategy_params["method"] != "risk_parity" else 1.0
        strategy_params["rf"]              = rf
        strategy_params["cov_estimator"]   = st.selectbox(
            "Covariance", ["ledoit_wolf", "oas", "sample"],
            format_func={"ledoit_wolf": "Ledoit-Wolf", "oas": "OAS", "sample": "Sample"}.get,
        )
        strategy_params["mean_estimator"]  = st.selectbox(
            "Returns", ["james_stein", "sample"],
            format_func={"james_stein": "James-Stein", "sample": "Sample"}.get,
        )
        strategy_params["min_obs"] = 60
    else:
        st.write("No parameters.")

# ── Validation ────────────────────────────────────────────────────────────────

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers.")
    st.stop()

if strategy_key == "ma_crossover" and strategy_params.get("fast", 0) >= strategy_params.get("slow", 1):
    st.error("Fast window must be shorter than slow window.")
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────

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
    st.error("Need at least 2 tickers with overlapping price data.")
    st.stop()

spy_series: pd.Series | None = None
if add_spy:
    spy_raw = _fetch("SPY", start_date, end_date)
    if not spy_raw.empty:
        spy_aligned = spy_raw.reindex(price_df.index).ffill().bfill()
        spy_series = spy_aligned.dropna() if not spy_aligned.dropna().empty else None

# ── Strategy factory helper ───────────────────────────────────────────────────

def _make_strategy(name: str, params: dict):
    if name == "buy_and_hold":
        return strategies.buy_and_hold()
    if name == "ma_crossover":
        return strategies.ma_crossover(params["fast"], params["slow"])
    if name == "momentum":
        return strategies.cross_sectional_momentum(
            params["lookback_months"], params["skip_months"], params["top_k"],
        )
    return strategies.walk_forward_optimizer(
        lookback_months=params["lookback_months"],
        method=params["method"],
        rf=params["rf"],
        weight_cap=params["weight_cap"],
        cov_estimator=params["cov_estimator"],
        mean_estimator=params["mean_estimator"],
        min_obs=params["min_obs"],
    )


# ── Run backtests (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _run(
    price_data: pd.DataFrame,
    strategy_name: str,
    params: dict,
    rebal_freq: str,
    tx_cost_bps: float,
) -> bt.BacktestResult:
    return bt.run_backtest(
        price_data, _make_strategy(strategy_name, params),
        rebalance_freq=rebal_freq, cost_bps=tx_cost_bps,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _run_ew(price_data: pd.DataFrame, rebal_freq: str) -> bt.BacktestResult:
    return bt.run_backtest(
        price_data, strategies.buy_and_hold(),
        rebalance_freq=rebal_freq, cost_bps=0.0,
    )


spinner_msg = (
    "Running walk-forward backtest (re-optimizing at each rebalance, ~30 s first run)..."
    if strategy_key == "walk_forward"
    else "Running backtest..."
)
with st.spinner(spinner_msg):
    result    = _run(price_df, strategy_key, strategy_params, rebalance_freq, float(cost_bps))
    ew_result = _run_ew(price_df, rebalance_freq)

# ── IS / OOS segmentation ─────────────────────────────────────────────────────

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

# ── Section 1: Equity curve ───────────────────────────────────────────────────

st.header("Equity curve")


def _norm100(s: pd.Series) -> pd.Series:
    return s / s.iloc[0] * 100


def _ts_ms(ts) -> int:
    """Integer milliseconds since epoch for Plotly shapes on date axes."""
    return int(pd.Timestamp(ts).timestamp() * 1000)


eq_fig = go.Figure()

if first_active > result.equity.index[0]:
    eq_fig.add_vrect(
        x0=_ts_ms(result.equity.index[0]),
        x1=_ts_ms(first_active),
        fillcolor=NEUTRAL, opacity=0.15, line_width=0,
        annotation_text="Burn-in", annotation_position="top left",
    )

eq_fig.add_vline(
    x=_ts_ms(split_ts),
    line_dash="dot", line_color=NEUTRAL, line_width=1.5,
    annotation_text="IS / OOS", annotation_position="top right",
)

eq_fig.add_trace(go.Scatter(
    x=result.equity.index, y=_norm100(result.equity),
    mode="lines", name=STRATEGY_LABELS[strategy_key],
    line=dict(color=PRIMARY, width=2),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
))

eq_fig.add_trace(go.Scatter(
    x=ew_result.equity.index, y=_norm100(ew_result.equity),
    mode="lines", name="Equal weight B&H",
    line=dict(color=POSITIVE, width=1.5, dash="dash"),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
))

if spy_series is not None:
    eq_fig.add_trace(go.Scatter(
        x=spy_series.index, y=_norm100(spy_series),
        mode="lines", name="SPY (raw)",
        line=dict(color=BENCHMARK, width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
    ))

eq_fig.update_layout(
    yaxis_type="log",
    yaxis_title="Portfolio value (log scale, indexed to 100)",
    height=440,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    legend=dict(x=0.02, y=0.98),
)
apply_chart_theme(eq_fig)
st.plotly_chart(eq_fig, use_container_width=True)
st.caption(
    "The shaded region marks the burn-in period during which the strategy is "
    "warming up; it is excluded from in-sample performance metrics. "
    "The vertical dashed line is the IS/OOS split date. "
    "The equal-weight benchmark is modeled with zero transaction costs."
)

# ── Section 2: Drawdown ───────────────────────────────────────────────────────

st.header("Drawdown")


def _dd_series(equity: pd.Series) -> pd.Series:
    rets = equity.pct_change().dropna()
    return analysis.drawdown(rets).series * 100 if not rets.empty else pd.Series(dtype=float)


dd_fig = go.Figure()

if first_active > result.equity.index[0]:
    dd_fig.add_vrect(
        x0=_ts_ms(result.equity.index[0]),
        x1=_ts_ms(first_active),
        fillcolor=NEUTRAL, opacity=0.15, line_width=0,
    )

dd_fig.add_vline(x=_ts_ms(split_ts), line_dash="dot", line_color=NEUTRAL, line_width=1.5)

strat_dd = _dd_series(result.equity.loc[first_active:])
ew_dd    = _dd_series(ew_result.equity.loc[first_active:])

dd_fig.add_trace(go.Scatter(
    x=strat_dd.index, y=strat_dd,
    mode="lines", name=STRATEGY_LABELS[strategy_key],
    line=dict(color=PRIMARY, width=2),
    fill="tozeroy", fillcolor=PRIMARY_10,
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
))

dd_fig.add_trace(go.Scatter(
    x=ew_dd.index, y=ew_dd,
    mode="lines", name="Equal weight B&H",
    line=dict(color=POSITIVE, width=1.5, dash="dash"),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
))

dd_fig.update_layout(
    yaxis_title="Drawdown (%)",
    height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    legend=dict(x=0.02, y=0.05),
)
apply_chart_theme(dd_fig)
st.plotly_chart(dd_fig, use_container_width=True)

# ── Section 3: IS / OOS performance table ────────────────────────────────────

st.header("Performance: IS vs OOS")

if len(is_equity) < 5:
    st.warning(
        "The in-sample period contains very few observations after the burn-in window. "
        "Move the IS/OOS split date later, or reduce the strategy's warm-up requirement, "
        "to obtain reliable in-sample statistics."
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

perf_table = pd.DataFrame({
    f"IS — {STRATEGY_LABELS[strategy_key]}":  _fmt_stats(is_stats),
    f"OOS — {STRATEGY_LABELS[strategy_key]}": _fmt_stats(oos_stats),
    "IS — Equal weight":                      _fmt_stats(ew_is_stats),
    "OOS — Equal weight":                     _fmt_stats(ew_oos_stats),
})
st.dataframe(perf_table, use_container_width=True)

# Overfitting signal
is_sharpe  = is_stats.get("Sharpe",  float("nan"))
oos_sharpe = oos_stats.get("Sharpe", float("nan"))
if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe):
    if is_sharpe > 0 and oos_sharpe < 0:
        st.error(
            f"The out-of-sample Sharpe ({oos_sharpe:.2f}) is negative while the "
            f"in-sample Sharpe ({is_sharpe:.2f}) was positive. This is a strong "
            "indicator of overfitting or a structural regime change between the two periods."
        )
    elif is_sharpe > 0.5 and oos_sharpe < is_sharpe * 0.5:
        st.warning(
            f"The out-of-sample Sharpe ({oos_sharpe:.2f}) is less than half the "
            f"in-sample Sharpe ({is_sharpe:.2f}). The strategy's parameters may not "
            "generalize beyond the fitted period."
        )

# ── Section 4: Trade log ──────────────────────────────────────────────────────

n_trades = len(result.trade_log)
with st.expander(f"Trade log ({n_trades} rebalance{'s' if n_trades != 1 else ''})"):
    if result.trade_log.empty:
        st.info("The strategy held cash for the entire period; no rebalances were recorded.")
    else:
        display_log = result.trade_log.copy()
        display_log["turnover"] = display_log["turnover"].map("{:.1%}".format)
        display_log["cost_pct"] = display_log["cost_pct"].map("{:.3%}".format)
        st.dataframe(display_log, use_container_width=True)

# ── Section 5: Weight history heatmap ────────────────────────────────────────

with st.expander("Weight history"):
    wh = result.weights_history
    if wh.empty:
        st.info("No rebalances recorded.")
    else:
        wh_pct = (wh * 100).round(1)
        wh_fig = go.Figure(go.Heatmap(
            z=wh_pct.values,
            x=wh_pct.columns.tolist(),
            y=[str(d.date()) if hasattr(d, "date") else str(d) for d in wh_pct.index],
            colorscale="Blues",
            zmin=0, zmax=100,
            text=wh_pct.values,
            texttemplate="%{text:.0f}%",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        ))
        wh_fig.update_layout(
            title="Target weights at each rebalance (%)",
            height=max(300, 22 * len(wh_pct)),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        apply_chart_theme(wh_fig)
        st.plotly_chart(wh_fig, use_container_width=True)
