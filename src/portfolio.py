"""
Portfolio optimization primitives.

Three decisions worth understanding:

1. max_sharpe uses the Markowitz/Lintner transform (y = w·κ) to turn the
   non-convex Sharpe maximization into a convex QP. Key identity:
     max (μ−rf)'w / sqrt(w'Σw)  ≡  min y'Σy  s.t. (μ−rf)'y = 1, y ≥ 0
   We recover weights as w = y / Σyᵢ. Per-asset weight cap becomes:
     yᵢ ≤ weight_cap · Σyⱼ  (linear in y, so still a QP).

2. risk_parity uses a log-barrier: minimize 0.5·w'Σw − Σlog(wᵢ).
   At the optimum, first-order conditions give wᵢ·(Σw)ᵢ = const for all i,
   which is exactly equal risk contribution. Solved via L-BFGS-B (scipy)
   because cvxpy's DCP rules don't accept log as a minimization objective here.

3. efficient_frontier sweeps target returns from the global min-variance return
   to the max individual-asset return, solving a parametric QP at each step.
   All cvxpy solves use CLARABEL (bundled with cvxpy ≥ 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


@dataclass
class PortfolioResult:
    weights: pd.Series      # ticker → weight, sums to 1.0
    expected_return: float  # annualized
    volatility: float       # annualized
    sharpe: float
    method: str


def expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Geometric annualized mean return per ticker."""
    n = len(returns)
    return (1 + returns).prod() ** (TRADING_DAYS / n) - 1


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    return returns.cov() * TRADING_DAYS


def portfolio_stats(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    method: str = "custom",
) -> PortfolioResult:
    """Compute annualized return, vol, and Sharpe for any weight vector."""
    w = weights.reindex(mu.index).fillna(0.0).values
    ret = float(mu.values @ w)
    vol = float(np.sqrt(np.clip(w @ cov.values @ w, 0.0, None)))
    sharpe = (ret - rf) / vol if vol > 1e-12 else float("nan")
    return PortfolioResult(
        weights=pd.Series(w, index=mu.index),
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        method=method,
    )


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    weight_cap: float = 1.0,
) -> PortfolioResult:
    """
    Maximize Sharpe ratio via the Markowitz/Lintner transform (convex QP).
    Raises ValueError if all expected returns are at or below rf.
    """
    excess = mu.values - rf
    if np.all(excess <= 0):
        raise ValueError(
            "All expected returns are at or below the risk-free rate; "
            "max Sharpe is undefined. Lower rf or choose a different method."
        )

    n = len(mu)
    Sigma = cov.values + 1e-8 * np.eye(n)  # small ridge → positive definite
    y = cp.Variable(n, nonneg=True)

    constraints = [excess @ y == 1]
    if weight_cap < 1.0:
        # wᵢ = yᵢ/Σyⱼ ≤ weight_cap  ⟺  yᵢ ≤ weight_cap · Σyⱼ  (linear in y)
        constraints.append(y <= weight_cap * cp.sum(y))

    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sigma)), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate") or y.value is None:
        raise RuntimeError(f"max_sharpe: solver returned {prob.status!r}")

    raw = np.clip(y.value, 0.0, None)
    weights = pd.Series(raw / raw.sum(), index=mu.index)
    return portfolio_stats(weights, mu, cov, rf, method="max_sharpe")


def min_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    weight_cap: float = 1.0,
) -> PortfolioResult:
    """Minimize portfolio variance (long-only, optional per-asset weight cap)."""
    n = len(cov)
    Sigma = cov.values + 1e-8 * np.eye(n)

    w = cp.Variable(n, nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Sigma)),
        [cp.sum(w) == 1, w <= weight_cap],
    )
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise RuntimeError(f"min_variance: solver returned {prob.status!r}")

    raw = np.clip(w.value, 0.0, None)
    weights = pd.Series(raw / raw.sum(), index=cov.index)
    return portfolio_stats(weights, mu, cov, rf, method="min_variance")


def risk_parity(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
) -> PortfolioResult:
    """
    Equal risk contribution via log-barrier minimization (L-BFGS-B).
    Unconstrained: each asset contributes equally to total portfolio variance.
    Low-vol assets receive more capital — that's the point, not a bug.
    """
    n = len(cov)
    Sigma = cov.values

    def obj(w: np.ndarray) -> float:
        return 0.5 * float(w @ Sigma @ w) - float(np.sum(np.log(w)))

    def grad(w: np.ndarray) -> np.ndarray:
        return Sigma @ w - 1.0 / w

    result = minimize(
        obj, np.ones(n) / n, jac=grad, method="L-BFGS-B",
        bounds=[(1e-8, None)] * n,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
    )

    raw = np.clip(result.x, 0.0, None)
    weights = pd.Series(raw / raw.sum(), index=cov.index)
    return portfolio_stats(weights, mu, cov, rf, method="risk_parity")


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 40,
    weight_cap: float = 1.0,
) -> pd.DataFrame:
    """
    Mean-variance efficient frontier via parametric QP sweep.
    Returns DataFrame[volatility, expected_return] in annualized decimal units.
    Sweeps from the global min-variance return to the max individual-asset return.
    """
    n = len(mu)
    Sigma = cov.values + 1e-8 * np.eye(n)

    mv = min_variance(mu, cov, rf=0.0, weight_cap=weight_cap)
    mu_min = mv.expected_return
    mu_max = float(mu.max())

    records = []
    for target in np.linspace(mu_min, mu_max, n_points):
        w = cp.Variable(n, nonneg=True)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, mu.values @ w == target, w <= weight_cap],
        )
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            vol = float(np.sqrt(np.clip(w.value @ Sigma @ w.value, 0.0, None)))
            records.append({"volatility": vol, "expected_return": target})

    return pd.DataFrame(records)
