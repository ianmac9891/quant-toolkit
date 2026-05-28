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
    """Spot prices where P&L = 0 at max-DTE expiration (linear interpolation)."""
    max_dte = max((l.dte for l in legs if l.option_type != "stock"), default=365)
    S_range = np.linspace(S0 * 0.30, S0 * 2.50, n_points)
    pnl     = position_pnl(legs, S_range, S0, r, q, days_elapsed=max_dte)

    bes = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:
            slope = (pnl[i + 1] - pnl[i]) / (S_range[i + 1] - S_range[i])
            be    = S_range[i] - pnl[i] / slope
            bes.append(round(float(be), 2))
    return bes


def prob_of_profit(
    legs: List[Leg], S0: float, r: float, q: float = 0.0, n_sims: int = 10_000,
) -> float:
    """Risk-neutral Monte Carlo POP at position's max DTE."""
    max_dte = max((l.dte for l in legs if l.option_type != "stock"), default=365)
    T       = max_dte / 365.0
    avg_iv  = float(np.mean([l.iv for l in legs if l.option_type not in ("stock",)]) or 0.20)

    rng = np.random.default_rng(42)
    z   = rng.standard_normal(n_sims)
    log_S = np.log(S0) + (r - q - 0.5 * avg_iv ** 2) * T + avg_iv * np.sqrt(T) * z
    S_term = np.exp(log_S)

    pnl = position_pnl(legs, S_term, S0, r, q, days_elapsed=max_dte)
    return float((pnl > 0).mean())


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

    return float(sigma) if 0 < sigma < 20 else None


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
