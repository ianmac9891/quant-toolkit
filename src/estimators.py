"""
Robust estimators for portfolio mean and covariance.

Covariance estimators:
- sample_covariance: standard sample estimator; unreliable when n_assets ≈ n_obs
- ledoit_wolf_covariance: analytical shrinkage toward scaled identity (Ledoit & Wolf 2004)
- oas_covariance: Oracle Approximating Shrinkage (Chen et al. 2010); lower error than LW
  on most real-asset universes

Mean estimators:
- sample_mean: geometric annualized mean; high variance on short windows
- james_stein_mean: shrinks each asset's estimate toward the cross-sectional grand mean,
  reducing the impact of extreme sample estimates

Michaud resampling:
- resampled_weights: iid bootstrap × n_bootstrap → optimize each → average weights.
  Smooths the optimizer's tendency to concentrate heavily in a few assets when
  estimation error is high.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

from src import portfolio as pf

TRADING_DAYS = 252


# ── Covariance estimators ─────────────────────────────────────────────────────

def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard sample covariance, annualized."""
    return returns.cov() * TRADING_DAYS


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf analytical shrinkage toward scaled identity, annualized."""
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(
        lw.covariance_ * TRADING_DAYS,
        index=returns.columns,
        columns=returns.columns,
    )


def oas_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Oracle Approximating Shrinkage estimator, annualized."""
    oas = OAS().fit(returns.values)
    return pd.DataFrame(
        oas.covariance_ * TRADING_DAYS,
        index=returns.columns,
        columns=returns.columns,
    )


COV_ESTIMATORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "sample":      sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "oas":         oas_covariance,
}


# ── Mean estimators ───────────────────────────────────────────────────────────

def sample_mean(returns: pd.DataFrame) -> pd.Series:
    """Geometric annualized mean return per ticker."""
    n = len(returns)
    return (1 + returns).prod() ** (TRADING_DAYS / n) - 1


def james_stein_mean(returns: pd.DataFrame, shrinkage: float = 0.5) -> pd.Series:
    """
    James-Stein estimator: shrinks each asset's sample mean toward the grand mean.

    μ_shrunk = (1 − w) · μ_sample + w · μ̄ · 1
    where μ̄ = mean(μ_sample) is the equal-weighted grand mean.

    # TODO: replace fixed shrinkage with the data-driven JS intensity:
    # w = min(1, (k − 2) · σ² / ‖μ − μ̄·1‖²)
    # where k = number of assets and σ² = mean(diag(Σ)) / T is the average
    # variance of the individual mean estimates. Requires estimating Σ first.
    # Fixed 0.5 is a reasonable conservative default for now.
    """
    mu = sample_mean(returns)
    grand_mean = float(mu.mean())
    return (1.0 - shrinkage) * mu + shrinkage * grand_mean


MEAN_ESTIMATORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "sample":      sample_mean,
    "james_stein": james_stein_mean,
}


# ── Michaud resampling ────────────────────────────────────────────────────────

def resampled_weights(
    returns: pd.DataFrame,
    method: str,
    rf: float,
    weight_cap: float,
    cov_estimator: str,
    mean_estimator: str,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> pd.Series:
    """
    Michaud-style resampling: iid bootstrap × n_bootstrap, optimize each sample,
    average the resulting weight vectors.

    Resamples the single optimal portfolio for the chosen method — not the full
    frontier (200 × 40 QPs = 8,000 solves would be too slow for interactive use).
    The same estimator pair is applied on each bootstrap sample for consistency.
    Failed bootstrap samples (rare solver failures) are skipped via nanmean.
    """
    rng = np.random.default_rng(random_state)
    T = len(returns)
    tickers = returns.columns
    k = len(tickers)

    cov_fn = COV_ESTIMATORS[cov_estimator]
    mu_fn = MEAN_ESTIMATORS[mean_estimator]
    weight_matrix = np.full((n_bootstrap, k), np.nan)

    for i in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        sample = returns.iloc[idx].copy()
        sample.columns = tickers

        try:
            mu_i = mu_fn(sample)
            cov_i = cov_fn(sample)

            if method == "max_sharpe":
                res = pf.max_sharpe(mu_i, cov_i, rf=rf, weight_cap=weight_cap)
            elif method == "min_variance":
                res = pf.min_variance(mu_i, cov_i, rf=rf, weight_cap=weight_cap)
            else:
                res = pf.risk_parity(mu_i, cov_i, rf=rf)

            weight_matrix[i] = res.weights.reindex(tickers).fillna(0).values
        except Exception:
            pass  # infeasible bootstrap sample — leave row as nan, skipped by nanmean

    mean_w = np.nanmean(weight_matrix, axis=0)
    mean_w = np.clip(mean_w, 0.0, None)
    mean_w /= mean_w.sum()
    return pd.Series(mean_w, index=tickers)
