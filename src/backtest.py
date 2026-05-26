"""
Backtesting engine.

No-lookahead convention: strategy_fn receives prices.iloc[:i+1] — all data up to
and including the current bar. Weights returned apply starting the *next* bar.
On day 0 a forced rebalance sets the initial allocation; the first earned return
is on day 1.

Weight drift: between rebalances the engine updates current_weights after each
day's return, so turnover measured at the next rebalance reflects the actual
drifted composition, not the prior target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import analysis

TRADING_DAYS = 252

StrategyFn = Callable[[pd.DataFrame], Optional[pd.Series]]

_FREQ_MAP = {"W": "W", "M": "ME", "Q": "QE"}


@dataclass
class BacktestResult:
    equity:          pd.Series       # daily equity curve, indexed by trading date
    returns:         pd.Series       # daily simple returns
    weights_history: pd.DataFrame    # target weights at each rebalance
    trade_log:       pd.DataFrame    # date, turnover, cost_pct, n_assets
    tickers:         list[str]


def _normalize(w: pd.Series) -> pd.Series:
    total = w.sum()
    return w if total < 1e-12 else w / total


def _rebal_schedule(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """Last actual trading date within each calendar period."""
    if freq == "D":
        return set(index)
    # Series whose values are the timestamps themselves; .last() returns the
    # last observed timestamp in each period bucket.
    dummy = pd.Series(index.tolist(), index=index)
    return set(dummy.resample(_FREQ_MAP[freq]).last().dropna())


def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    initial_capital: float = 10_000.0,
    rebalance_freq: str = "M",
    cost_bps: float = 10.0,
    min_history: int = 1,
) -> BacktestResult:
    """
    Parameters
    ----------
    prices          : adj_close DataFrame; columns = tickers, index = trading dates
    strategy_fn     : receives prices.iloc[:i+1]; returns target weights or None/empty for cash
    initial_capital : starting portfolio value
    rebalance_freq  : "D" / "W" / "M" / "Q"
    cost_bps        : one-way transaction cost in bps applied on turnover
    min_history     : minimum bars before engine calls strategy_fn
    """
    prices = prices.ffill().dropna(how="all")
    tickers = list(prices.columns)
    n = len(prices)

    rebal_dates = _rebal_schedule(prices.index, rebalance_freq)
    rebal_dates.add(prices.index[0])

    equity_arr = np.empty(n)
    equity_arr[0] = initial_capital
    current_weights = pd.Series(0.0, index=tickers)

    weight_rows: list[tuple[pd.Timestamp, pd.Series]] = []
    trade_rows:  list[dict] = []

    for i in range(n):
        date = prices.index[i]

        if i > 0:
            ret = (prices.iloc[i] / prices.iloc[i - 1] - 1.0).fillna(0.0)
            port_ret = float((current_weights * ret).sum())
            equity_arr[i] = equity_arr[i - 1] * (1.0 + port_ret)

            if current_weights.sum() > 1e-12:
                current_weights = _normalize(current_weights * (1.0 + ret))

        if date in rebal_dates and i >= min_history - 1:
            target = strategy_fn(prices.iloc[: i + 1])
            if target is not None and not target.empty and target.sum() > 1e-12:
                target = _normalize(target.reindex(tickers).fillna(0.0))
                turnover = float((target - current_weights).abs().sum() / 2.0)
                cost = turnover * cost_bps / 10_000.0
                equity_arr[i] *= 1.0 - cost
                current_weights = target.copy()
                trade_rows.append({
                    "date":     date,
                    "turnover": turnover,
                    "cost_pct": cost,
                    "n_assets": int((target > 1e-6).sum()),
                })
                weight_rows.append((date, current_weights.copy()))

    equity  = pd.Series(equity_arr, index=prices.index, name="equity")
    returns = equity.pct_change().fillna(0.0)

    if weight_rows:
        wh = pd.DataFrame(
            [w.values for _, w in weight_rows],
            index=[d for d, _ in weight_rows],
            columns=tickers,
        )
        wh.index.name = "date"
    else:
        wh = pd.DataFrame(columns=tickers)

    trade_log = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(columns=["date", "turnover", "cost_pct", "n_assets"])
    )

    return BacktestResult(
        equity=equity,
        returns=returns,
        weights_history=wh,
        trade_log=trade_log,
        tickers=tickers,
    )


def perf_stats(
    equity: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
    rf: float = 0.0,
) -> dict:
    """Key risk/return metrics for an equity curve segment."""
    rets = equity.pct_change().dropna()
    nan = float("nan")
    keys = ["Ann. return", "Ann. vol", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Avg daily turnover"]
    if len(rets) < 2:
        return {k: nan for k in keys}

    dd      = analysis.drawdown(rets)
    ann_ret = analysis.annualized_return(rets)
    calmar  = ann_ret / abs(dd.max_drawdown) if dd.max_drawdown < -1e-10 else nan

    avg_to = (
        trade_log["turnover"].sum() / len(rets)
        if trade_log is not None and not trade_log.empty and "turnover" in trade_log.columns
        else nan
    )

    return {
        "Ann. return":       ann_ret,
        "Ann. vol":          analysis.annualized_volatility(rets),
        "Sharpe":            analysis.sharpe_ratio(rets, rf=rf),
        "Sortino":           analysis.sortino_ratio(rets, rf=rf),
        "Max drawdown":      dd.max_drawdown,
        "Calmar":            calmar,
        "Avg daily turnover": avg_to,
    }
