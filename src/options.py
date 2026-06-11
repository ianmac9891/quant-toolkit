"""
Black-Scholes pricing and Greeks for European options on a dividend-paying underlying.

Conventions
-----------
- Theta:  per calendar day (annual / 365)
- Vega:   per 1 vol point (per 0.01 change in sigma)
- Rho:    per 1 percentage point (per 0.01 change in r)
- Greeks are signed by direction (long +, short –) and scaled by quantity × 100
- A leg with option_type="stock" represents 100 shares: delta=1, gamma=theta=vega=rho=0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats

CONTRACT_MULTIPLIER = 100


# ── Black-Scholes core ────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    """(d1, d2) for Black-Scholes-Merton with continuous dividend yield q."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(opt: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return np.nan
    disc = np.exp(-r * T)
    fwd  = S * np.exp(-q * T)
    if opt == "call":
        return fwd * stats.norm.cdf(d1) - K * disc * stats.norm.cdf(d2)
    return K * disc * stats.norm.cdf(-d2) - fwd * stats.norm.cdf(-d1)


def bs_delta(opt: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return (1.0 if S > K else 0.0) if opt == "call" else (-1.0 if S < K else 0.0)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return np.nan
    eq = np.exp(-q * T)
    return eq * stats.norm.cdf(d1) if opt == "call" else eq * (stats.norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return np.nan
    return np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(opt: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Annual theta / 365 (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return np.nan
    fwd  = S * np.exp(-q * T)
    disc = np.exp(-r * T)
    base = -(fwd * stats.norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if opt == "call":
        annual = base - r * K * disc * stats.norm.cdf(d2) + q * fwd * stats.norm.cdf(d1)
    else:
        annual = base + r * K * disc * stats.norm.cdf(-d2) - q * fwd * stats.norm.cdf(-d1)
    return annual / 365.0


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega per 0.01 change in sigma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d1):
        return np.nan
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T) * 0.01


def bs_rho(opt: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Rho per 0.01 change in r."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if np.isnan(d2):
        return np.nan
    disc = np.exp(-r * T)
    if opt == "call":
        return K * T * disc * stats.norm.cdf(d2) * 0.01
    return -K * T * disc * stats.norm.cdf(-d2) * 0.01


# ── Position structures ───────────────────────────────────────────────────────

@dataclass
class Leg:
    option_type: str   # "call", "put", or "stock"
    direction: str     # "long" or "short"
    strike: float      # ignored for "stock"
    dte: int           # days to expiry; ignored for "stock"
    iv: float          # annualized, e.g. 0.30; ignored for "stock"
    quantity: int = 1  # number of contracts (1 contract = 100 shares)


@dataclass
class LegResult:
    leg: Leg
    entry_price: float    # BS price per share at entry
    price: float          # current BS price per share
    delta: float          # per share (unsigned)
    gamma: float
    theta: float
    vega: float
    rho: float
    # Position-level (signed, scaled by qty × 100)
    pos_delta: float
    pos_gamma: float
    pos_theta: float
    pos_vega: float
    pos_rho: float


def _stock_result(leg: Leg, S: float, S0: float) -> LegResult:
    sign = 1 if leg.direction == "long" else -1
    mult = sign * leg.quantity * CONTRACT_MULTIPLIER
    return LegResult(
        leg=leg,
        entry_price=S0, price=S,
        delta=1.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        pos_delta=float(mult), pos_gamma=0.0,
        pos_theta=0.0, pos_vega=0.0, pos_rho=0.0,
    )


def price_leg(leg: Leg, S: float, S0: float, r: float, q: float = 0.0, days_elapsed: int = 0) -> LegResult:
    """Price one leg; days_elapsed is calendar days since trade entry."""
    if leg.option_type == "stock":
        return _stock_result(leg, S, S0)

    T_entry = leg.dte / 365.0
    T_now   = max(leg.dte - days_elapsed, 0) / 365.0
    sign    = 1 if leg.direction == "long" else -1
    mult    = sign * leg.quantity * CONTRACT_MULTIPLIER

    entry_p = bs_price(leg.option_type, S0, leg.strike, T_entry, r, leg.iv, q)
    cur_p   = bs_price(leg.option_type, S,  leg.strike, T_now,   r, leg.iv, q)
    d       = bs_delta(leg.option_type, S,  leg.strike, T_now,   r, leg.iv, q)
    g       = bs_gamma(S, leg.strike, T_now, r, leg.iv, q)
    th      = bs_theta(leg.option_type, S, leg.strike, T_now, r, leg.iv, q)
    ve      = bs_vega(S, leg.strike, T_now, r, leg.iv, q)
    rh      = bs_rho(leg.option_type, S, leg.strike, T_now, r, leg.iv, q)

    return LegResult(
        leg=leg, entry_price=entry_p, price=cur_p,
        delta=d, gamma=g, theta=th, vega=ve, rho=rh,
        pos_delta=mult * d, pos_gamma=mult * g,
        pos_theta=mult * th, pos_vega=mult * ve, pos_rho=mult * rh,
    )


# ── P&L ───────────────────────────────────────────────────────────────────────

def position_pnl(
    legs: List[Leg],
    S_range: np.ndarray,
    S0: float,
    r: float,
    q: float = 0.0,
    days_elapsed: int = 0,
) -> np.ndarray:
    """
    P&L at each spot in S_range relative to position entry cost at S0.
    P&L = (current_value - entry_value) per position.
    """
    pnl = np.zeros(len(S_range), dtype=float)
    for leg in legs:
        sign = 1 if leg.direction == "long" else -1
        mult = sign * leg.quantity * CONTRACT_MULTIPLIER

        if leg.option_type == "stock":
            entry_p = S0
            cur_p   = S_range
        else:
            T_entry = leg.dte / 365.0
            T_now   = max(leg.dte - days_elapsed, 0) / 365.0
            entry_p = bs_price(leg.option_type, S0, leg.strike, T_entry, r, leg.iv, q)
            cur_p   = np.array([bs_price(leg.option_type, s, leg.strike, T_now, r, leg.iv, q)
                                 for s in S_range])

        pnl += mult * (cur_p - entry_p)
    return pnl


def breakevens(
    legs: List[Leg], S0: float, r: float, q: float = 0.0, n_points: int = 2000,
) -> List[float]:
    """Spot prices where P&L = 0 at the front expiration (linear interpolation)."""
    horizon = eval_horizon_dte(legs)
    S_range = np.linspace(S0 * 0.30, S0 * 2.50, n_points)
    pnl     = position_pnl(legs, S_range, S0, r, q, days_elapsed=horizon)

    bes = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:
            slope = (pnl[i + 1] - pnl[i]) / (S_range[i + 1] - S_range[i])
            be    = S_range[i] - pnl[i] / slope
            bes.append(round(float(be), 2))
    return bes


def atm_leg_iv(legs: List[Leg], S0: float) -> float:
    """IV of the option leg whose strike is nearest the spot — the default
    proxy for the single underlying volatility used in simulation."""
    option_legs = [l for l in legs if l.option_type != "stock" and l.iv > 0]
    if not option_legs:
        return 0.20
    nearest = min(option_legs, key=lambda l: abs(l.strike - S0))
    return float(nearest.iv)


def eval_horizon_dte(legs: List[Leg]) -> int:
    """Evaluation horizon for expiry metrics: the FRONT (earliest) option
    expiration. Valuing the position past the front expiry would require
    path-dependent settlement of the expired legs, which a terminal-price
    model cannot represent."""
    return min((l.dte for l in legs if l.option_type != "stock"), default=365)


def prob_of_profit(
    legs: List[Leg],
    S0: float,
    r: float,
    q: float = 0.0,
    n_sims: int = 20_000,
    underlying_vol: Optional[float] = None,
) -> float:
    """Risk-neutral Monte Carlo POP at the front expiration.

    The underlying is simulated with a single volatility (the underlying has
    one diffusion regardless of how many legs reference it). Defaults to the
    IV of the leg nearest the money; pass underlying_vol to override.
    """
    horizon = eval_horizon_dte(legs)
    T       = horizon / 365.0
    vol     = float(underlying_vol) if underlying_vol else atm_leg_iv(legs, S0)

    rng = np.random.default_rng(42)
    z   = rng.standard_normal(n_sims)
    log_S = np.log(S0) + (r - q - 0.5 * vol ** 2) * T + vol * np.sqrt(T) * z
    S_term = np.exp(log_S)

    pnl = position_pnl(legs, S_term, S0, r, q, days_elapsed=horizon)
    return float((pnl > 0).mean())


@dataclass
class PayoffBounds:
    max_profit: float          # at the front expiration; +inf if unbounded
    max_loss: float            # most negative P&L; -inf if unbounded
    profit_unbounded: bool
    loss_unbounded: bool
    upper_slope: float         # dP&L/dS per $1 of spot as S → ∞ (per position)


def payoff_bounds(
    legs: List[Leg], S0: float, r: float, q: float = 0.0,
) -> PayoffBounds:
    """Analytic max profit / max loss at the front expiration.

    Unboundedness is determined from the asymptotic payoff slope as S → ∞:
    each stock leg and each call (any expiry — deep ITM calls converge to
    forward delta ≈ 1) contributes its signed share count; puts contribute
    nothing on the upside. The downside is always bounded because the spot
    is floored at zero. Bounded extremes are then read off a dense terminal
    grid spanning [≈0, 4×S0] plus every strike.
    """
    horizon = eval_horizon_dte(legs)

    upper_slope = 0.0
    for leg in legs:
        mult = (1 if leg.direction == "long" else -1) * leg.quantity * CONTRACT_MULTIPLIER
        if leg.option_type in ("stock", "call"):
            upper_slope += mult

    strikes = [l.strike for l in legs if l.option_type != "stock"]
    grid = np.unique(np.concatenate([
        np.linspace(0.01, max(S0 * 4.0, (max(strikes) if strikes else S0) * 1.5), 2000),
        np.array(strikes, dtype=float) if strikes else np.array([S0]),
    ]))
    pnl = position_pnl(legs, grid, S0, r, q, days_elapsed=horizon)

    profit_unbounded = upper_slope > 1e-9
    loss_unbounded   = upper_slope < -1e-9

    max_profit = float("inf") if profit_unbounded else float(np.nanmax(pnl))
    max_loss   = float("-inf") if loss_unbounded else float(np.nanmin(pnl))

    return PayoffBounds(
        max_profit=max_profit,
        max_loss=max_loss,
        profit_unbounded=profit_unbounded,
        loss_unbounded=loss_unbounded,
        upper_slope=upper_slope,
    )


# ── Implied volatility solver ─────────────────────────────────────────────────

def implied_vol(
    opt: str, market_price: float,
    S: float, K: float, T: float, r: float, q: float = 0.0,
    tol: float = 1e-7, max_iter: int = 200,
) -> Optional[float]:
    """Newton-Raphson IV. Returns None if it cannot converge."""
    if T <= 0 or market_price <= 0:
        return None

    intrinsic = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    if market_price <= intrinsic:
        return None

    # Brenner-Subrahmanyam initial guess
    sigma = max(np.sqrt(2 * np.pi / T) * market_price / S, 0.01)

    for _ in range(max_iter):
        price = bs_price(opt, S, K, T, r, sigma, q)
        vega  = bs_vega(S, K, T, r, sigma, q) / 0.01   # un-scale: raw vega per unit sigma
        if abs(vega) < 1e-12:
            break
        sigma -= (price - market_price) / vega
        sigma  = max(sigma, 1e-6)
        if abs(bs_price(opt, S, K, T, r, sigma, q) - market_price) < tol:
            return float(sigma)

    if 0 < sigma < 20 and abs(bs_price(opt, S, K, T, r, sigma, q) - market_price) < tol:
        return float(sigma)

    # Newton failed (deep ITM/OTM: vanishing vega). Bisection fallback — the BS
    # price is monotone in sigma, so this converges whenever a root exists.
    lo, hi = 1e-4, 20.0
    p_lo = bs_price(opt, S, K, T, r, lo, q)
    p_hi = bs_price(opt, S, K, T, r, hi, q)
    if not (p_lo <= market_price <= p_hi):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(opt, S, K, T, r, mid, q)
        if abs(p_mid - market_price) < tol:
            return float(mid)
        if p_mid < market_price:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ── Templates ─────────────────────────────────────────────────────────────────

def template_legs(name: str, S0: float) -> List[dict]:
    """Return a list of raw dicts suitable for constructing a legs DataFrame."""
    K    = round(S0)
    w    = round(S0 * 0.05)   # 5% wing width

    templates = {
        "Long Call": [
            dict(option_type="call", direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
        ],
        "Long Put": [
            dict(option_type="put",  direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
        ],
        "Covered Call": [
            dict(option_type="stock", direction="long", strike=K,     dte=0,  iv=0.0,  quantity=1),
            dict(option_type="call",  direction="short", strike=K+w,  dte=30, iv=0.30, quantity=1),
        ],
        "Bull Call Spread": [
            dict(option_type="call", direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
            dict(option_type="call", direction="short", strike=K+w,   dte=30, iv=0.27, quantity=1),
        ],
        "Bear Put Spread": [
            dict(option_type="put",  direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
            dict(option_type="put",  direction="short", strike=K-w,   dte=30, iv=0.27, quantity=1),
        ],
        "Long Butterfly (Calls)": [
            dict(option_type="call", direction="long",  strike=K-w,   dte=30, iv=0.30, quantity=1),
            dict(option_type="call", direction="short", strike=K,     dte=30, iv=0.28, quantity=2),
            dict(option_type="call", direction="long",  strike=K+w,   dte=30, iv=0.30, quantity=1),
        ],
        "Long Straddle": [
            dict(option_type="call", direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
            dict(option_type="put",  direction="long",  strike=K,     dte=30, iv=0.30, quantity=1),
        ],
        "Long Strangle": [
            dict(option_type="call", direction="long",  strike=K+w,   dte=30, iv=0.30, quantity=1),
            dict(option_type="put",  direction="long",  strike=K-w,   dte=30, iv=0.30, quantity=1),
        ],
        "Calendar Spread": [
            dict(option_type="call", direction="short", strike=K,     dte=30, iv=0.30, quantity=1),
            dict(option_type="call", direction="long",  strike=K,     dte=60, iv=0.30, quantity=1),
        ],
    }
    return templates.get(name, [])
