"""Reference-value tests for the event-study statistics.

The fixture is constructed so the OLS market model recovers its parameters
exactly: the synthetic residuals are chosen orthogonal to the market return
and zero-mean, which makes alpha, beta, sigma_e, and every downstream
statistic hand-derivable. Expected values are computed step by step from the
defining formulas in this file rather than asserted as opaque constants.
"""

import numpy as np
import pandas as pd
import pytest

from src import event_study as es

# ── Fixture construction ──────────────────────────────────────────────────────
# Estimation window: n = 8 sessions.
# Market returns alternate +a, -a so their mean is exactly zero.
# Residuals follow the pattern +e, +e, -e, -e repeated: zero-mean, and their
# dot product with the market pattern is e*a - e*a - e*a + e*a = 0 per block,
# so OLS recovers alpha and beta with no estimation error in the coefficients.

ALPHA, BETA = 0.001, 1.2
A, E = 0.01, 0.002

MKT_EST = np.array([+A, -A, +A, -A, +A, -A, +A, -A])
RESID_EST = np.array([+E, +E, -E, -E, +E, +E, -E, -E])
STOCK_EST = ALPHA + BETA * MKT_EST + RESID_EST

# Event window: one session before, the event session, one after.
MKT_EVT = np.array([0.005, -0.02, 0.01])
AR_EVT = np.array([0.001, 0.03, -0.004])   # chosen abnormal returns
STOCK_EVT = ALPHA + BETA * MKT_EVT + AR_EVT


def _build_series():
    """ret_t, ret_m with a business-day index: 8 estimation sessions, a
    2-session buffer of exactly-on-model returns, then the 3-session event
    window centered on the event date."""
    buffer_mkt = np.array([0.0, 0.0])
    buffer_stock = ALPHA + BETA * buffer_mkt   # zero residual in the buffer
    mkt = np.concatenate([MKT_EST, buffer_mkt, MKT_EVT])
    stock = np.concatenate([STOCK_EST, buffer_stock, STOCK_EVT])
    idx = pd.bdate_range("2024-01-02", periods=len(mkt))
    return pd.Series(stock, index=idx), pd.Series(mkt, index=idx)


# Event date = the middle session of the event window (position 11: 8
# estimation + 2 buffer + 1 pre-event session).
def _event_date(ret):
    return ret.index[11].date()


def test_market_model_recovers_parameters_exactly():
    ret_t, ret_m = _build_series()
    # Window arithmetic: the series runs [8 estimation][2 buffer][pre,
    # event, post]. _fit_market_model ends the estimation window at the
    # session `buffer_days` back from the last pre-event session, counting
    # inclusively, so buffer_days=4 (2 buffer sessions + the pre-event
    # session + the inclusive boundary) lands the 8-session OLS window
    # exactly on the constructed estimation block.
    fit = es._fit_market_model(ret_t, ret_m, _event_date(ret_t),
                               estimation_days=8, buffer_days=4)
    assert fit.alpha == pytest.approx(ALPHA, abs=1e-12)
    assert fit.beta == pytest.approx(BETA, abs=1e-12)

    # sigma_e: ss_res = 8 e^2, divided by n - 2 = 6
    expected_sigma_e = np.sqrt(8 * E**2 / 6)
    assert fit.sigma_e == pytest.approx(expected_sigma_e, rel=1e-9)

    # Patell inputs: market mean is 0 by construction, ss is 8 a^2
    assert fit.mkt_mean == pytest.approx(0.0, abs=1e-15)
    assert fit.mkt_ss == pytest.approx(8 * A**2, rel=1e-12)


def test_abnormal_returns_and_car():
    ret_t, ret_m = _build_series()
    fit = es._fit_market_model(ret_t, ret_m, _event_date(ret_t),
                               estimation_days=8, buffer_days=4)
    times, ar, car, actual, predicted, mkt = es._event_window(
        ret_t, ret_m, _event_date(ret_t), fit, pre_event=1, post_event=1)

    assert list(times) == [-1, 0, 1]
    # The fit is exact, so abnormal returns equal the constructed AR_EVT
    np.testing.assert_allclose(ar, AR_EVT, atol=1e-12)
    np.testing.assert_allclose(car, np.cumsum(AR_EVT), atol=1e-12)
    np.testing.assert_allclose(mkt, MKT_EVT, atol=1e-15)


def test_patell_statistics_match_formula():
    ret_t, ret_m = _build_series()
    fit = es._fit_market_model(ret_t, ret_m, _event_date(ret_t),
                               estimation_days=8, buffer_days=4)
    _, ar, _, _, _, mkt = es._event_window(
        ret_t, ret_m, _event_date(ret_t), fit, pre_event=1, post_event=1)

    z, p, scar = es._patell_stats(fit, ar, mkt)

    # Derive the expected values from the Patell definitions:
    # C_t = 1 + 1/n + (R_mt - mean)^2 / ss, SAR_t = AR_t / (sigma_e sqrt(C_t)),
    # Z = sum(SAR) / sqrt(L (n-2)/(n-4)), SCAR = sum(AR) / (sigma_e sqrt(sum C_t))
    n, L = 8, 3
    c_t = 1 + 1 / n + (mkt - fit.mkt_mean) ** 2 / fit.mkt_ss
    sar = ar / (fit.sigma_e * np.sqrt(c_t))
    expected_z = sar.sum() / np.sqrt(L * (n - 2) / (n - 4))
    expected_scar = ar.sum() / (fit.sigma_e * np.sqrt(c_t.sum()))

    assert z == pytest.approx(expected_z, rel=1e-12)
    assert scar == pytest.approx(expected_scar, rel=1e-12)
    assert 0.0 <= p <= 1.0


def test_bmp_t_statistic_is_cross_sectional_t():
    # Three events with SCARs 2.0, 1.0, 1.5:
    # mean = 1.5, sample std = 0.5, t = 1.5 / (0.5 / sqrt(3)) = 3 sqrt(3)
    t, p = es._bmp_stats([2.0, 1.0, 1.5])
    assert t == pytest.approx(1.5 / (0.5 / np.sqrt(3)), rel=1e-12)
    assert 0.0 < p < 1.0


def test_bmp_requires_cross_section():
    t, p = es._bmp_stats([1.7])
    assert np.isnan(t) and np.isnan(p)


def test_aggregate_patell_scales_by_sqrt_n():
    # Z_agg = (1.0 + 2.0) / sqrt(2)
    z, p = es._aggregate_patell([1.0, 2.0, float("nan")])
    assert z == pytest.approx(3.0 / np.sqrt(2), rel=1e-12)
    assert 0.0 < p < 1.0
