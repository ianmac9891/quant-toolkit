"""
Cross-asset correlation analytics.

Functions operate on a daily simple-returns DataFrame (columns = tickers).
Correlations are computed on overlapping observations; rolling measures use a
fixed window of trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample pairwise Pearson correlation."""
    return returns.corr()


def mean_offdiag_correlation(corr: pd.DataFrame) -> float:
    """Average of the off-diagonal entries of a correlation matrix."""
    vals = corr.values
    n = vals.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(vals[mask]))


def rolling_mean_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Average pairwise correlation through time — the diversification pulse.

    Rises toward 1 in stress episodes, when assets sell off together and
    diversification fails exactly when it is needed.
    """
    n = returns.shape[1]
    if n < 2:
        return pd.Series(dtype=float)
    roll = returns.rolling(window).corr()
    # roll has a (date, ticker) MultiIndex; average the off-diagonals per date
    mask_sum = []
    dates = []
    for dt, mat in roll.groupby(level=0):
        m = mat.droplevel(0).values
        if np.isnan(m).all():
            continue
        offdiag = m[~np.eye(n, dtype=bool)]
        if np.isnan(offdiag).all():
            continue
        dates.append(dt)
        mask_sum.append(float(np.nanmean(offdiag)))
    return pd.Series(mask_sum, index=pd.DatetimeIndex(dates), name="mean_corr")


def rolling_pair_correlation(
    returns: pd.DataFrame, a: str, b: str, window: int = 63
) -> pd.Series:
    """Rolling correlation between two columns."""
    return returns[a].rolling(window).corr(returns[b]).dropna()


def pc1_variance_share(corr: pd.DataFrame) -> float:
    """Share of total variance explained by the first principal component of
    the correlation matrix — a concentration gauge. Near 1/N indicates well-
    spread risk; values approaching 1 indicate one common factor drives the
    universe."""
    vals = corr.dropna(axis=0, how="all").dropna(axis=1, how="all").values
    if vals.shape[0] < 2 or np.isnan(vals).any():
        return float("nan")
    eig = np.linalg.eigvalsh(vals)
    return float(eig[-1] / eig.sum())


def diversification_ratio(returns: pd.DataFrame, weights: pd.Series | None = None) -> float:
    """Weighted average asset volatility divided by portfolio volatility
    (Choueifaty & Coignard). Equal weights by default. 1.0 means no
    diversification benefit; higher is better."""
    n = returns.shape[1]
    if n < 2:
        return float("nan")
    w = (weights.reindex(returns.columns).fillna(0.0).values
         if weights is not None else np.full(n, 1.0 / n))
    vols = returns.std(ddof=1).values
    cov = returns.cov().values
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol <= 0:
        return float("nan")
    return float((w @ vols) / port_vol)


def extreme_pairs(corr: pd.DataFrame, k: int = 3) -> tuple[list[tuple], list[tuple]]:
    """(highest, lowest) k pairs as (ticker_a, ticker_b, corr), excluding the
    diagonal and duplicate orderings."""
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda p: p[2])
    return pairs[-k:][::-1], pairs[:k]
