"""Reference-value tests for the Black-Scholes-Merton engine.

Benchmarks come from Hull, Options, Futures, and Other Derivatives (the
worked examples that appear across editions), so a regression here means the
engine disagrees with the standard textbook treatment, not with itself.
"""

import numpy as np
import pytest

from src import options as op


# ── Hull Ch. 15 worked example: S=42, K=40, r=10%, sigma=20%, T=0.5 ───────────
# Published values: call 4.76, put 0.81.

HULL_PRICE = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=0.0)


def test_hull_call_price():
    c = op.bs_price("call", HULL_PRICE["S"], HULL_PRICE["K"], HULL_PRICE["T"],
                    HULL_PRICE["r"], HULL_PRICE["sigma"], HULL_PRICE["q"])
    assert c == pytest.approx(4.76, abs=0.01)


def test_hull_put_price():
    p = op.bs_price("put", HULL_PRICE["S"], HULL_PRICE["K"], HULL_PRICE["T"],
                    HULL_PRICE["r"], HULL_PRICE["sigma"], HULL_PRICE["q"])
    assert p == pytest.approx(0.81, abs=0.01)


# ── Hull Ch. 19 Greeks example: S=49, K=50, r=5%, sigma=20%, T=0.3846 ─────────
# Published values: delta 0.522, gamma 0.066, vega 12.1 (per unit sigma),
# theta -4.31 (per year), rho 8.91 (per unit r). Our conventions: theta is
# per calendar day (annual / 365); vega and rho are per 0.01 move.

HULL_GREEKS = dict(S=49.0, K=50.0, T=0.3846, r=0.05, sigma=0.20, q=0.0)


def test_hull_delta():
    d = op.bs_delta("call", **HULL_GREEKS)
    assert d == pytest.approx(0.522, abs=0.002)


def test_hull_gamma():
    g = op.bs_gamma(HULL_GREEKS["S"], HULL_GREEKS["K"], HULL_GREEKS["T"],
                    HULL_GREEKS["r"], HULL_GREEKS["sigma"], HULL_GREEKS["q"])
    assert g == pytest.approx(0.066, abs=0.001)


def test_hull_vega():
    v = op.bs_vega(HULL_GREEKS["S"], HULL_GREEKS["K"], HULL_GREEKS["T"],
                   HULL_GREEKS["r"], HULL_GREEKS["sigma"], HULL_GREEKS["q"])
    # Hull reports 12.1 per unit of sigma; ours is per 0.01
    assert v * 100 == pytest.approx(12.1, abs=0.05)


def test_hull_theta():
    th = op.bs_theta("call", **HULL_GREEKS)
    # Hull reports -4.31 per year; ours is per calendar day
    assert th * 365 == pytest.approx(-4.31, abs=0.02)


def test_hull_rho():
    rh = op.bs_rho("call", **HULL_GREEKS)
    # Hull reports 8.91 per unit of r; ours is per 0.01
    assert rh * 100 == pytest.approx(8.91, abs=0.05)


# ── Hull index-option example with a dividend yield ───────────────────────────
# European call on an index: S=930, K=900, r=8%, q=3%, sigma=20%, T=2/12.
# Published value: 51.83.

def test_hull_dividend_yield_call():
    c = op.bs_price("call", 930.0, 900.0, 2.0 / 12.0, 0.08, 0.20, 0.03)
    assert c == pytest.approx(51.83, abs=0.05)


# ── Additional parameter sets: put-call parity across a grid ──────────────────
# c - p = S e^{-qT} - K e^{-rT} holds identically for European options, so any
# violation is a pricing bug rather than a tolerance question.

@pytest.mark.parametrize("S", [50.0, 100.0, 180.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.10, 0.45])
@pytest.mark.parametrize("q", [0.0, 0.025])
def test_put_call_parity_grid(S, K, T, sigma, q):
    r = 0.04
    c = op.bs_price("call", S, K, T, r, sigma, q)
    p = op.bs_price("put", S, K, T, r, sigma, q)
    parity = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(parity, abs=1e-9)


# ── Implied volatility round-trip ─────────────────────────────────────────────
# Price an option at a known sigma, solve the IV back, and require recovery to
# 1e-6. The deep-OTM case exercises the bisection fallback where Newton's
# vega is too small to converge.

@pytest.mark.parametrize("opt,S,K,T,r,q,sigma", [
    ("call", 100.0, 100.0, 0.5, 0.045, 0.0, 0.27),
    ("put", 100.0, 95.0, 0.25, 0.045, 0.0, 0.42),
    ("call", 930.0, 900.0, 2.0 / 12.0, 0.08, 0.03, 0.20),
    ("put", 100.0, 60.0, 1.0, 0.02, 0.01, 0.55),    # deep OTM put
    ("call", 100.0, 160.0, 0.75, 0.03, 0.0, 0.35),  # deep OTM call
])
def test_implied_vol_round_trip(opt, S, K, T, r, q, sigma):
    price = op.bs_price(opt, S, K, T, r, sigma, q)
    recovered = op.implied_vol(opt, price, S, K, T, r, q)
    assert recovered is not None
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_below_arbitrage_bound_returns_none():
    # A call cannot be worth less than its zero-volatility lower bound
    # S e^{-qT} - K e^{-rT}; asking for an IV there must fail cleanly.
    S, K, T, r, q = 150.0, 100.0, 0.25, 0.05, 0.0
    lower_bound = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert op.implied_vol("call", lower_bound - 0.5, S, K, T, r, q) is None
