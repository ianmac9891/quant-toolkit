"""
GARCH(1,1) volatility forecasting.

Fit   : arch_model on daily log returns × 100 (% units) for numerical stability.
Simulate : bootstrap — resample empirical standardized residuals, so fat tails
           are preserved without assuming a parametric shock distribution.
Drift    : applied in log-return space, so zero drift → flat median price path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model   # type: ignore

TRADING_DAYS = 252
_PCT = 100.0   # scaling factor: log returns × 100 for GARCH fitting


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GarchFit:
    # Parameters — omega in (%/day)^2 units; alpha, beta, gamma dimensionless
    omega: float
    alpha: float
    beta: float
    persistence: float          # alpha + beta (+ gamma/2 for GJR); must be < 1
    mu_pct: float               # fitted mean log return (% per day)
    gamma: float                # GJR asymmetry term; 0 for symmetric GARCH
    model: str                  # "garch" | "gjr"

    # Variance state — all in (%/day)^2
    h_current_pct2: float       # sigma_t^2 at last observation
    h_next_pct2: float          # sigma_{t+1}^2 = omega + alpha*eps_t^2 + beta*h_t
    h_lr_pct2: float            # omega / (1 - persistence) — long-run variance

    # Annualized vols, decimal fractions (e.g. 0.20 = 20%)
    current_ann_vol: float
    longrun_ann_vol: float
    vol_percentile: float       # 0–1: where current vol sits in fitted vol history
    vol_regime: str             # "elevated" | "normal" | "compressed"

    # Fit quality
    aic: float
    loglikelihood: float

    # Bootstrap inputs — standardized residuals from the fitted model
    std_resid: np.ndarray       # shape (T,); resampled for simulation


@dataclass
class VolForecast:
    current_price: float
    horizon_days: int
    drift_annual: float

    # Percentile price paths — each shape (horizon+1,), index 0 = current_price
    p2_5:  np.ndarray
    p10:   np.ndarray
    p25:   np.ndarray
    p50:   np.ndarray   # median; flat at current_price when drift_annual = 0
    p75:   np.ndarray
    p90:   np.ndarray
    p97_5: np.ndarray

    # Full terminal distribution — for O(1) P(> target) queries on every rerun
    terminal_prices: np.ndarray   # shape (n_sim,)

    fit: GarchFit


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_garch(prices: pd.Series, model_type: str = "garch") -> GarchFit:
    """
    Fit GARCH(1,1) or GJR-GARCH(1,1,1) to daily log returns.

    model_type "gjr" adds the asymmetry term γ·ε²·I(ε<0), capturing the
    leverage effect (volatility rises more after negative returns) that is
    characteristic of equity series. Persistence for GJR is α + β + γ/2
    (the indicator's expectation is 1/2 under symmetric shocks).

    Raises
    ------
    ValueError
        If persistence >= 1 (non-stationary) or the fit fails to converge.
        Caller should catch this and show a user-friendly message.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    rets_pct = log_rets * _PCT

    o = 1 if model_type == "gjr" else 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model  = arch_model(rets_pct, vol="Garch", p=1, o=o, q=1,
                            dist="Normal", mean="Constant")
        result = model.fit(disp="off", show_warning=False)

    p = result.params
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    beta  = float(p["beta[1]"])
    gamma = float(p.get("gamma[1]", 0.0))
    mu    = float(p.get("mu", p.get("Const", 0.0)))
    persistence = alpha + beta + gamma / 2.0

    if persistence >= 1.0:
        raise ValueError(
            f"Non-stationary fit: persistence = {persistence:.4f} ≥ 1. "
            "Try a longer fit window or a different ticker."
        )

    h_lr_pct2 = omega / (1.0 - persistence)

    sigma_t       = float(result.conditional_volatility.iloc[-1])   # % per day
    h_current_pct2 = sigma_t ** 2

    # One-step-ahead variance: omega + alpha*eps² (+ gamma*eps²·I(eps<0)) + beta*h
    last_resid    = float(result.resid.iloc[-1])   # in % units
    h_next_pct2   = (
        omega
        + alpha * last_resid ** 2
        + gamma * last_resid ** 2 * (1.0 if last_resid < 0 else 0.0)
        + beta * h_current_pct2
    )
    h_next_pct2   = max(h_next_pct2, 1e-8)         # numerical guard

    def _ann(h: float) -> float:
        return np.sqrt(h * TRADING_DAYS) / _PCT

    current_ann_vol = _ann(h_current_pct2)
    longrun_ann_vol = _ann(h_lr_pct2)

    # Vol regime: percentile of current vol in the full conditional-vol history
    cond_ann_vols = result.conditional_volatility.values / _PCT * np.sqrt(TRADING_DAYS)
    vol_pct       = float((cond_ann_vols < current_ann_vol).mean())
    if   vol_pct > 0.75: regime = "elevated"
    elif vol_pct < 0.25: regime = "compressed"
    else:                 regime = "normal"

    std_resid = result.std_resid.dropna().values.astype(float)

    return GarchFit(
        omega=omega, alpha=alpha, beta=beta,
        gamma=gamma, model=model_type,
        persistence=persistence, mu_pct=mu,
        h_current_pct2=h_current_pct2,
        h_next_pct2=h_next_pct2,
        h_lr_pct2=h_lr_pct2,
        current_ann_vol=current_ann_vol,
        longrun_ann_vol=longrun_ann_vol,
        vol_percentile=vol_pct,
        vol_regime=regime,
        aic=float(result.aic),
        loglikelihood=float(result.loglikelihood),
        std_resid=std_resid,
    )


# ── Analytic vol path ─────────────────────────────────────────────────────────

def analytic_vol_path(fit: GarchFit, horizon: int) -> np.ndarray:
    """
    Annualized vol at each forward step k = 1 … horizon (shape: (horizon,)).

    Starts from h_{t+1} (one-step-ahead forecast) and mean-reverts toward
    the long-run level:
        h_{t+k} = h_lr + persistence^(k-1) × (h_{t+1} − h_lr)

    Converges to longrun_ann_vol as horizon → ∞; speed set by persistence.
    """
    k      = np.arange(1, horizon + 1, dtype=float)
    h_path = fit.h_lr_pct2 + fit.persistence ** (k - 1) * (fit.h_next_pct2 - fit.h_lr_pct2)
    return np.sqrt(h_path * TRADING_DAYS) / _PCT   # annualized, decimal


# ── Bootstrap simulation ──────────────────────────────────────────────────────

def simulate_paths(
    fit: GarchFit,
    current_price: float,
    horizon: int,
    drift_annual: float = 0.0,
    n_sim: int = 10_000,
    seed: int = 42,
) -> VolForecast:
    """
    Bootstrap GARCH price-path simulation in log-return space.

    Residuals are resampled from the empirical distribution — no normality
    assumption. With drift_annual = 0, the median price path is flat at
    current_price (zero-drift in log space means no volatility drag).
    """
    rng = np.random.default_rng(seed)

    # Daily log-drift in % units
    drift_log_pct = np.log1p(drift_annual) / TRADING_DAYS * _PCT

    # Bootstrap: sample all residuals upfront — shape (n_sim, horizon)
    idx = rng.integers(0, len(fit.std_resid), size=(n_sim, horizon))
    Z   = fit.std_resid[idx]   # (n_sim, horizon)

    # Initialize: log-prices and GARCH variance
    log_prices = np.empty((n_sim, horizon + 1))
    log_prices[:, 0] = np.log(current_price)
    h = np.full(n_sim, fit.h_next_pct2)   # one-step-ahead variance as starting point

    for t in range(horizon):
        # Log return in % units: drift + GARCH shock
        r_pct = drift_log_pct + np.sqrt(h) * Z[:, t]        # (n_sim,)
        log_prices[:, t + 1] = log_prices[:, t] + r_pct / _PCT

        # Variance update — centered residual = shock component only.
        # The gamma term (zero for symmetric GARCH) loads only on negative shocks.
        eps_pct = r_pct - drift_log_pct                      # = sqrt(h) * Z[:, t]
        h = (
            fit.omega
            + fit.alpha * eps_pct ** 2
            + fit.gamma * eps_pct ** 2 * (eps_pct < 0)
            + fit.beta * h
        )

    price_paths = np.exp(log_prices)   # (n_sim, horizon+1)

    pcts = np.percentile(price_paths, [2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5], axis=0)

    return VolForecast(
        current_price=current_price,
        horizon_days=horizon,
        drift_annual=drift_annual,
        p2_5=pcts[0], p10=pcts[1], p25=pcts[2], p50=pcts[3],
        p75=pcts[4],  p90=pcts[5], p97_5=pcts[6],
        terminal_prices=price_paths[:, -1],
        fit=fit,
    )


# ── Probability query ─────────────────────────────────────────────────────────

def p_above(forecast: VolForecast, target: float) -> float:
    """P(terminal price > target). O(1) lookup from cached terminal_prices."""
    return float((forecast.terminal_prices > target).mean())
