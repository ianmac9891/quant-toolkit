"""Tests for analysis module. Run with: pytest tests/"""

import numpy as np
import pandas as pd
import pytest

from src import analysis


@pytest.fixture
def sample_prices():
    """A deterministic price series with known properties."""
    rng = np.random.default_rng(42)
    n = 252 * 3  # 3 years
    daily_log_rets = rng.normal(loc=0.0005, scale=0.012, size=n)  # ~12.6% ann return, ~19% ann vol
    prices = 100 * np.exp(daily_log_rets.cumsum())
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="price")


def test_simple_returns_length(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    assert len(rets) == len(sample_prices) - 1


def test_simple_vs_log_returns_close(sample_prices):
    # For small returns, simple ~ log
    simple = analysis.simple_returns(sample_prices)
    log = analysis.log_returns(sample_prices)
    assert (simple - log).abs().mean() < 0.0002


def test_cumulative_returns_endpoint(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    wealth = analysis.cumulative_returns(rets)
    # Wealth at end should equal price ratio
    expected = sample_prices.iloc[-1] / sample_prices.iloc[0]
    assert abs(wealth.iloc[-1] - expected) < 1e-9


def test_annualized_vol_in_expected_range(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    vol = analysis.annualized_volatility(rets)
    # We set daily stdev to 0.012, so ann vol ~ 0.012 * sqrt(252) ~= 0.19
    assert 0.15 < vol < 0.23


def test_sharpe_nonzero(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    s = analysis.sharpe_ratio(rets, rf=0.0)
    assert not np.isnan(s)


def test_sharpe_handles_zero_vol():
    rets = pd.Series([0.001] * 100)
    s = analysis.sharpe_ratio(rets)
    assert np.isnan(s)


def test_drawdown_negative(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    dd = analysis.drawdown(rets)
    assert dd.max_drawdown <= 0
    assert dd.peak_date <= dd.trough_date


def test_var_below_mean(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    var = analysis.historical_var(rets, 0.05)
    # 5th percentile must be less than the mean for any non-degenerate distribution
    assert var < rets.mean()


def test_cvar_worse_than_var(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    var = analysis.historical_var(rets, 0.05)
    cvar = analysis.historical_cvar(rets, 0.05)
    assert cvar <= var  # CVaR is the avg of the tail, must be <= the threshold


def test_summary_table_runs(sample_prices):
    rets = analysis.simple_returns(sample_prices)
    summary = analysis.summary_table(rets, rf=0.045)
    assert len(summary) > 0
    assert "Sharpe ratio" in summary.index
