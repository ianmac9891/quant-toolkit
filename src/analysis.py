"""
Analysis primitives.

Two decisions worth understanding:

1. Simple vs log returns. Simple returns are additive across assets (so
   portfolio returns = weighted sum of asset returns). Log returns are
   additive across time and better behaved statistically (closer to normal
   for short horizons, never goes below -100%). We compute both. Use
   simple returns when you're combining assets, log returns when you're
   doing statistics on a single time series.

2. Annualization factor. We assume 252 trading days/year. Variance scales
   linearly with time under iid assumption, so stdev scales with sqrt(T).
   That's why annualized vol = daily stdev * sqrt(252) and annualized
   return = daily mean * 252. The iid assumption is wrong in practice
   (autocorrelation exists) but it's the standard convention and what
   every Sharpe ratio you've ever seen uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# -------------------------------------------------------------------------
# Returns
# -------------------------------------------------------------------------

def simple_returns(prices: pd.Series) -> pd.Series:
    """Arithmetic returns: (p_t - p_{t-1}) / p_{t-1}."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns: ln(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: pd.Series, log: bool = False) -> pd.Series:
    """
    Wealth index assuming $1 invested at t=0.
    If log=True, treat input as log returns.
    """
    if log:
        return np.exp(returns.cumsum())
    return (1 + returns).cumprod()


# -------------------------------------------------------------------------
# Risk and performance metrics
# -------------------------------------------------------------------------

def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return from simple returns."""
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return total ** (1 / years) - 1 if years > 0 else float("nan")


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized stdev of returns."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualized Sharpe. rf is the annual risk-free rate (e.g. 0.05 for 5%).
    Excess return: subtract daily-equivalent rf from daily returns.
    """
    if returns.empty:
        return float("nan")
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    sigma = excess.std(ddof=1)
    if not np.isfinite(sigma) or sigma < 1e-12:
        return float("nan")
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Like Sharpe but only penalizes downside volatility (returns below rf).
    Argument for using this: investors don't actually dislike upside variance.
    Argument against: penalizing only downside throws away information and
    can be gamed by strategies with rare large losses.
    """
    if returns.empty:
        return float("nan")
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    sigma_down = downside.std(ddof=1)
    if not np.isfinite(sigma_down) or sigma_down < 1e-12:
        return float("nan")
    return float(excess.mean() / sigma_down * np.sqrt(TRADING_DAYS))


@dataclass
class DrawdownResult:
    series: pd.Series        # drawdown at each date (<=0)
    max_drawdown: float      # most negative value
    peak_date: pd.Timestamp  # date of peak before max DD
    trough_date: pd.Timestamp


def drawdown(returns: pd.Series) -> DrawdownResult:
    """
    Drawdown series: percentage decline from running max of wealth index.
    """
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    trough_date = dd.idxmin()
    peak_date = wealth.loc[:trough_date].idxmax()
    return DrawdownResult(
        series=dd,
        max_drawdown=float(dd.min()),
        peak_date=peak_date,
        trough_date=trough_date,
    )


# -------------------------------------------------------------------------
# Tail risk
# -------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical VaR at confidence (1 - alpha). Returns a negative number.
    Example: alpha=0.05 -> 5th percentile of daily return distribution.
    Interpretation: 'on 5% of days we expect to lose at least |VaR|'.
    """
    if returns.empty:
        return float("nan")
    return float(np.quantile(returns, alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional VaR / Expected Shortfall: average loss in the worst alpha
    tail. More informative than VaR because it tells you how bad the bad
    days actually are.
    """
    if returns.empty:
        return float("nan")
    var = historical_var(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else float("nan")


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    VaR assuming returns are normal. Almost always understates tail risk
    because real returns are fat-tailed. We compute it for comparison so
    you can see the gap vs historical.
    """
    if returns.empty:
        return float("nan")
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return float(mu + sigma * stats.norm.ppf(alpha))


# -------------------------------------------------------------------------
# Distribution diagnostics
# -------------------------------------------------------------------------

def distribution_stats(returns: pd.Series) -> dict:
    """Moments + normality test. Use this to gut-check return distributions."""
    if returns.empty:
        return {}
    jb_stat, jb_p = stats.jarque_bera(returns.values)
    return {
        "n_obs": int(len(returns)),
        "mean_daily": float(returns.mean()),
        "stdev_daily": float(returns.std(ddof=1)),
        "skewness": float(stats.skew(returns)),
        "kurtosis_excess": float(stats.kurtosis(returns)),  # excess (normal=0)
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),  # p < 0.05 -> reject normality
        "min": float(returns.min()),
        "max": float(returns.max()),
    }


# -------------------------------------------------------------------------
# Summary helper for the Streamlit UI
# -------------------------------------------------------------------------

def summary_table(returns: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    """One-row summary suitable for st.dataframe display."""
    dd = drawdown(returns)
    rows = {
        "Annualized return": annualized_return(returns),
        "Annualized volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, rf=rf),
        "Sortino ratio": sortino_ratio(returns, rf=rf),
        "Max drawdown": dd.max_drawdown,
        "Historical VaR (95%)": historical_var(returns, 0.05),
        "Historical CVaR (95%)": historical_cvar(returns, 0.05),
        "Skewness": float(stats.skew(returns)) if not returns.empty else float("nan"),
        "Excess kurtosis": float(stats.kurtosis(returns)) if not returns.empty else float("nan"),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])
