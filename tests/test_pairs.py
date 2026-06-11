"""Reference-value tests for the pairs engine on simulated processes.

The half-life check uses an Ornstein-Uhlenbeck (discrete AR(1)) spread with a
known persistence parameter, so the population half-life is exact and the
estimate must land within sampling tolerance. The cointegration checks build
one pair that is cointegrated by construction and one that is two independent
random walks; Engle-Granger should separate them cleanly at the chosen sample
size and seed.
"""

import numpy as np
import pandas as pd
import pytest

from src import pairs as pr

T = 4000
HEDGE = 0.8
PHI = 0.9   # AR(1) persistence of the spread
# Population half-life of an AR(1) with coefficient PHI:
# s_t = PHI s_{t-1} + eps, deviations decay as PHI^k, so the half-life is
# ln(0.5) / ln(PHI) sessions.
EXPECTED_HALF_LIFE = np.log(0.5) / np.log(PHI)   # about 6.58


def _cointegrated_pair(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=T)
    # Common stochastic trend: a random walk in logs
    log_b = np.cumsum(rng.normal(0, 0.01, T)) + np.log(50.0)
    # OU spread around the linear relationship
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = PHI * spread[t - 1] + rng.normal(0, 0.004)
    log_a = 0.10 + HEDGE * log_b + spread
    return (pd.Series(np.exp(log_a), index=idx),
            pd.Series(np.exp(log_b), index=idx))


def _independent_pair(seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=T)
    log_a = np.cumsum(rng.normal(0, 0.01, T)) + np.log(80.0)
    log_b = np.cumsum(rng.normal(0, 0.01, T)) + np.log(40.0)
    return (pd.Series(np.exp(log_a), index=idx),
            pd.Series(np.exp(log_b), index=idx))


def test_hedge_ratio_recovered():
    pa, pb = _cointegrated_pair()
    res = pr.analyze_pair(pa, pb, "A", "B")
    assert res.hedge_ratio == pytest.approx(HEDGE, abs=0.02)


def test_half_life_matches_ou_parameter():
    pa, pb = _cointegrated_pair()
    res = pr.analyze_pair(pa, pb, "A", "B")
    # Sampling error on 4000 observations keeps the AR(1) estimate close to
    # PHI; a quarter-relative band is comfortably outside noise while still
    # catching sign or unit errors in the half-life formula.
    assert np.isfinite(res.half_life_days)
    assert res.half_life_days == pytest.approx(EXPECTED_HALF_LIFE, rel=0.25)


def test_constructed_pair_is_cointegrated():
    pa, pb = _cointegrated_pair()
    res = pr.analyze_pair(pa, pb, "A", "B")
    assert res.coint_p < 0.01


def test_independent_walks_are_not_cointegrated():
    pa, pb = _independent_pair()
    res = pr.analyze_pair(pa, pb, "A", "B")
    assert res.coint_p > 0.05


def test_short_overlap_raises():
    idx = pd.bdate_range("2024-01-02", periods=60)
    s = pd.Series(np.linspace(100, 110, 60), index=idx)
    with pytest.raises(ValueError):
        pr.analyze_pair(s, s * 1.01, "A", "B")
