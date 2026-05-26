"""
Strategy factories for the backtesting engine.

Each factory returns a StrategyFn:
    (prices: pd.DataFrame) -> pd.Series   # target weights, sum ≈ 1

The function receives all prices up to the current bar (no lookahead) and must
return a non-negative weight vector that sums to ~1, or an empty Series to signal
staying in cash.

Strategies
----------
buy_and_hold             — fixed or equal weights, rebalances back to target
ma_crossover             — equal-weight assets whose fast MA is above slow MA
cross_sectional_momentum — top-k assets by trailing return (Jegadeesh-Titman)
walk_forward_optimizer   — rolling MVO / risk parity re-fit on recent history
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src import estimators as est
from src import portfolio as pf


def buy_and_hold(weights: Optional[pd.Series] = None):
    """
    Fixed-weight strategy. Returns the same weights every call.
    If weights is None, defaults to equal weight across all available tickers.
    """
    def strategy(prices: pd.DataFrame) -> pd.Series:
        if weights is not None:
            w = weights.reindex(prices.columns).fillna(0.0)
        else:
            w = pd.Series(1.0, index=prices.columns)
        total = w.sum()
        return w / total if total > 1e-12 else w

    return strategy


def ma_crossover(fast: int = 20, slow: int = 60):
    """
    Equal-weight the assets whose fast moving average is above their slow moving
    average. Returns an empty Series (cash) when no asset qualifies or when there
    is insufficient history.
    """
    def strategy(prices: pd.DataFrame) -> pd.Series:
        if len(prices) < slow:
            return pd.Series(dtype=float)
        fast_ma = prices.iloc[-fast:].mean()
        slow_ma = prices.iloc[-slow:].mean()
        active = fast_ma[fast_ma > slow_ma].index
        if active.empty:
            return pd.Series(dtype=float)
        w = pd.Series(1.0 / len(active), index=active)
        return w.reindex(prices.columns).fillna(0.0)

    return strategy


def cross_sectional_momentum(
    lookback_months: int = 12,
    skip_months: int = 1,
    top_k: int = 3,
):
    """
    Jegadeesh-Titman momentum: rank by return over [t − lookback, t − skip],
    go long the top_k equally. The skip avoids the short-term reversal that
    tends to reverse momentum at the one-month horizon.

    Uses ~21 trading days per month for day-count conversion.
    """
    lookback_days = int(lookback_months * 21)
    skip_days     = int(skip_months * 21)
    min_days      = lookback_days + skip_days

    def strategy(prices: pd.DataFrame) -> pd.Series:
        if len(prices) < min_days:
            return pd.Series(dtype=float)
        end   = len(prices) - skip_days
        start = end - lookback_days
        window = prices.iloc[start:end]
        ret    = (window.iloc[-1] / window.iloc[0] - 1).fillna(-1.0)
        top    = ret.nlargest(min(top_k, len(ret))).index
        w      = pd.Series(1.0 / len(top), index=top)
        return w.reindex(prices.columns).fillna(0.0)

    return strategy


def walk_forward_optimizer(
    lookback_months: int = 36,
    method: str = "max_sharpe",
    rf: float = 0.04,
    weight_cap: float = 1.0,
    cov_estimator: str = "ledoit_wolf",
    mean_estimator: str = "james_stein",
    min_obs: int = 60,
):
    """
    Rolling MVO / risk parity: re-fit on the most recent lookback_months of
    returns at each rebalance. Falls back to equal weight when the window has
    fewer than min_obs observations or the solver fails.
    """
    lookback_days = int(lookback_months * 21)
    cov_fn = est.COV_ESTIMATORS[cov_estimator]
    mu_fn  = est.MEAN_ESTIMATORS[mean_estimator]

    def strategy(prices: pd.DataFrame) -> pd.Series:
        rets = prices.iloc[-lookback_days:].pct_change().dropna()
        if len(rets) < min_obs:
            return pd.Series(1.0 / prices.shape[1], index=prices.columns)
        try:
            mu  = mu_fn(rets)
            cov = cov_fn(rets)
            if method == "max_sharpe":
                res = pf.max_sharpe(mu, cov, rf=rf, weight_cap=weight_cap)
            elif method == "min_variance":
                res = pf.min_variance(mu, cov, rf=rf, weight_cap=weight_cap)
            else:
                res = pf.risk_parity(mu, cov, rf=rf)
            return res.weights.reindex(prices.columns).fillna(0.0)
        except Exception:
            return pd.Series(1.0 / prices.shape[1], index=prices.columns)

    return strategy
