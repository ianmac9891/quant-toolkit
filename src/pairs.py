"""
Relative value (pairs) analysis — cointegration and spread diagnostics.

Method
------
1. Align the two price series and work in log prices.
2. OLS hedge ratio: log(A) = c + h·log(B) + ε. The residual ε is the spread.
3. Engle-Granger cointegration test (statsmodels `coint`) on the log prices —
   its p-value accounts for the estimated hedge ratio, which a plain ADF on
   the residual does not.
4. ADF test on the spread reported alongside for reference.
5. Half-life of mean reversion from an AR(1) on the spread:
   Δs_t = a + b·s_{t-1} + e_t  →  half-life = −ln(2)/ln(1 + b) for −1 < b < 0.
6. Spread z-score against the full-sample mean and standard deviation.

A cointegrated pair with a short half-life and an extreme current z-score is
the classical mean-reversion setup; none of this implies the relationship
persists out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

TRADING_DAYS = 252


@dataclass
class PairResult:
    ticker_a: str
    ticker_b: str
    n_obs: int

    hedge_ratio: float          # h in log(A) = c + h·log(B)
    intercept: float

    spread: pd.Series           # OLS residual (log-price units)
    zscore: pd.Series           # spread standardized by full-sample mean/std
    current_z: float

    coint_stat: float           # Engle-Granger test statistic
    coint_p: float              # Engle-Granger p-value
    adf_stat: float             # ADF on the spread (reference)
    adf_p: float

    half_life_days: float       # NaN if the spread is not mean-reverting
    return_corr: float          # daily simple-return correlation

    rebased_a: pd.Series        # prices rebased to 100 for overlay charts
    rebased_b: pd.Series


def analyze_pair(prices_a: pd.Series, prices_b: pd.Series,
                 ticker_a: str = "A", ticker_b: str = "B") -> PairResult:
    """Run the full pair diagnostic on two aligned daily price series.

    Raises ValueError if the overlapping history is too short to be meaningful.
    """
    joint = pd.concat(
        [prices_a.rename("a"), prices_b.rename("b")], axis=1, join="inner"
    ).dropna()
    if len(joint) < 120:
        raise ValueError(
            f"Only {len(joint)} overlapping sessions — at least 120 are required "
            "for a meaningful cointegration estimate."
        )

    log_a = np.log(joint["a"].values)
    log_b = np.log(joint["b"].values)
    n = len(joint)

    # OLS hedge ratio
    X = np.column_stack([np.ones(n), log_b])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
    intercept, hedge = float(coeffs[0]), float(coeffs[1])

    spread_vals = log_a - (intercept + hedge * log_b)
    spread = pd.Series(spread_vals, index=joint.index, name="spread")

    mu, sd = float(spread.mean()), float(spread.std(ddof=1))
    zscore = (spread - mu) / sd if sd > 0 else spread * 0.0

    # Engle-Granger (accounts for the estimated hedge ratio)
    try:
        cstat, cp, _ = coint(log_a, log_b)
        coint_stat, coint_p = float(cstat), float(cp)
    except Exception:
        coint_stat, coint_p = float("nan"), float("nan")

    # ADF on the spread (reference only — uses standard ADF critical values)
    try:
        adf = adfuller(spread_vals, autolag="AIC")
        adf_stat, adf_p = float(adf[0]), float(adf[1])
    except Exception:
        adf_stat, adf_p = float("nan"), float("nan")

    # Half-life via AR(1) on the spread
    s_lag = spread_vals[:-1]
    s_diff = np.diff(spread_vals)
    Xh = np.column_stack([np.ones(len(s_lag)), s_lag])
    bcoef, _, _, _ = np.linalg.lstsq(Xh, s_diff, rcond=None)
    b = float(bcoef[1])
    half_life = float(-np.log(2.0) / np.log(1.0 + b)) if -1.0 < b < 0.0 else float("nan")

    ret_corr = float(
        joint["a"].pct_change().corr(joint["b"].pct_change())
    )

    return PairResult(
        ticker_a=ticker_a, ticker_b=ticker_b, n_obs=n,
        hedge_ratio=hedge, intercept=intercept,
        spread=spread, zscore=zscore, current_z=float(zscore.iloc[-1]),
        coint_stat=coint_stat, coint_p=coint_p,
        adf_stat=adf_stat, adf_p=adf_p,
        half_life_days=half_life, return_corr=ret_corr,
        rebased_a=joint["a"] / joint["a"].iloc[0] * 100.0,
        rebased_b=joint["b"] / joint["b"].iloc[0] * 100.0,
    )
