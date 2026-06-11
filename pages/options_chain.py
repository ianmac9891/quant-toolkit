"""Options Chain Explorer — listed chains, implied volatility structure, positioning.

Data comes from Yahoo's delayed option chains via yfinance (one request per
expiry). Chain quality varies: zero-bid strikes, placeholder IVs near 1e-5,
and dead strikes with no open interest are endemic, so every analytic here
filters to strikes with open interest and a usable quote rather than plotting
whatever the feed returns.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import ui
from src import options as op
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE,
    CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Equity & Derivatives Research", "Options Chain Explorer",
    "Listed option chains from delayed Yahoo Finance data: implied volatility "
    "smile and term structure, the straddle-implied expected move, open "
    "interest and volume positioning, and the full quote table with model "
    "deltas.",
)

# ── Cached chain access ───────────────────────────────────────────────────────
# 900-second TTL: chains move intraday but this app is a research terminal,
# not an execution screen, and Yahoo rate-limits aggressive chain polling.

@st.cache_data(ttl=900, show_spinner=False)
def _expiries(ticker: str) -> list[str]:
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def _chain(ticker: str, expiry: str):
    """(calls, puts, spot, quote_time) for one expiry; empty frames on failure."""
    try:
        ch = yf.Ticker(ticker).option_chain(expiry)
        spot = None
        qtime = None
        underlying = getattr(ch, "underlying", None)
        if isinstance(underlying, dict):
            spot = underlying.get("regularMarketPrice")
            ts = underlying.get("regularMarketTime")
            if ts:
                qtime = datetime.fromtimestamp(int(ts))
        return ch.calls.copy(), ch.puts.copy(), spot, qtime
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), None, None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to strikes a person could actually trade against: open interest
    present, an implied vol that is not Yahoo's missing-value placeholder, and
    at least one live side of the quote. Adds a mid column."""
    if df.empty:
        return df
    out = df.copy()
    for col in ("bid", "ask", "impliedVolatility", "openInterest", "volume", "strike"):
        if col not in out.columns:
            out[col] = np.nan
    out["openInterest"] = out["openInterest"].fillna(0)
    out["volume"] = out["volume"].fillna(0)
    out = out[
        (out["openInterest"] > 0)
        & (out["impliedVolatility"].notna())
        & (out["impliedVolatility"] > 0.005)
        & ~((out["bid"].fillna(0) <= 0) & (out["ask"].fillna(0) <= 0))
    ].copy()
    out["mid"] = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2.0
    # One-sided quotes: fall back to the live side rather than averaging with zero
    one_sided = (out["bid"].fillna(0) <= 0) | (out["ask"].fillna(0) <= 0)
    out.loc[one_sided, "mid"] = out.loc[one_sided, ["bid", "ask"]].max(axis=1)
    return out.sort_values("strike").reset_index(drop=True)


def _atm_row(df: pd.DataFrame, spot: float) -> pd.Series | None:
    if df.empty:
        return None
    idx = (df["strike"] - spot).abs().idxmin()
    return df.loc[idx]


@st.cache_data(ttl=900, show_spinner=False)
def _atm_term_structure(ticker: str, expiries: tuple[str, ...], spot: float) -> pd.DataFrame:
    """ATM straddle IV per expiry. One chain request per expiry; the page
    caption discloses that cost and the 900s cache amortizes it."""
    rows = []
    for exp in expiries:
        calls, puts, _, _ = _chain(ticker, exp)
        c, p = _clean(calls), _clean(puts)
        c_atm, p_atm = _atm_row(c, spot), _atm_row(p, spot)
        ivs = [r["impliedVolatility"] for r in (c_atm, p_atm) if r is not None]
        if ivs:
            rows.append({"expiry": exp, "atm_iv": float(np.mean(ivs))})
    return pd.DataFrame(rows)


# ── Parameters ────────────────────────────────────────────────────────────────

with ui.panel("Parameters"):
    c1, c2, c3, c4 = st.columns([1, 1.3, 1.3, 1.1])
    with c1:
        ticker = st.text_input("Instrument", value=ui.get_default_ticker("SPY")).upper().strip()

    if not ticker:
        ui.banner("info", "Enter an instrument symbol to begin.")
        st.stop()

    expiries = _expiries(ticker)
    if not expiries:
        # Distinguish a bad symbol from a symbol with no listed options
        probe = ui.fetch_prices(ticker, date.today().replace(year=date.today().year - 1),
                                date.today())
        if probe.ok:
            ui.banner("info", f"<b>{ticker}</b> has no listed options on the provider feed.")
        else:
            ui.data_unavailable(f"{ticker}: {probe.error}")
        st.stop()

    with c2:
        expiry = st.selectbox(
            "Expiry", expiries,
            index=min(2, len(expiries) - 1),
            help="Listed expirations reported by the provider.",
        )
    with c3:
        strike_band = st.slider(
            "Strike Range (% of spot)", min_value=5, max_value=50, value=20, step=5,
            help="Smile, positioning, and table are restricted to strikes within "
                 "this band around the current underlying price.",
        )
    with c4:
        rf_pct = st.number_input(
            "Risk-Free Rate (% per annum)", min_value=0.0, max_value=20.0,
            value=ui.get_default_rf_pct(), step=0.25,
            help="Used for model deltas. Defaults from the Yield Curve Monitor's "
                 "current 13-week bill rate when that page has been visited.",
        )

ui.remember_ticker(ticker)
r = rf_pct / 100.0

# ── Pull and clean the selected chain ─────────────────────────────────────────

with st.spinner(f"Retrieving {ticker} {expiry} chain..."):
    raw_calls, raw_puts, spot, qtime = _chain(ticker, expiry)

if raw_calls.empty and raw_puts.empty:
    ui.data_unavailable(f"{ticker} {expiry}: empty chain")
    st.stop()

if spot is None or not np.isfinite(spot) or spot <= 0:
    probe = ui.fetch_prices(ticker, date.today().replace(year=date.today().year - 1),
                            date.today())
    if probe.ok and "adj_close" in probe.df.columns:
        spot = float(probe.df["adj_close"].dropna().iloc[-1])
    else:
        ui.data_unavailable(f"{ticker}: no underlying quote")
        st.stop()
spot = float(spot)

calls = _clean(raw_calls)
puts  = _clean(raw_puts)

if calls.empty and puts.empty:
    ui.banner("info",
              f"The {expiry} chain has no strikes with open interest and a live "
              "quote. Choose a nearer expiry or a more liquid underlying.")
    st.stop()

lo, hi = spot * (1 - strike_band / 100), spot * (1 + strike_band / 100)
calls_band = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
puts_band  = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]

dte = max((pd.Timestamp(expiry).date() - date.today()).days, 0)
T = max(dte, 0.5) / 365.0   # same-day expiries: half a day rather than zero

# ── Headline ──────────────────────────────────────────────────────────────────

c_atm = _atm_row(calls, spot)
p_atm = _atm_row(puts, spot)

atm_ivs = [x["impliedVolatility"] for x in (c_atm, p_atm) if x is not None]
atm_iv = float(np.mean(atm_ivs)) if atm_ivs else float("nan")

straddle_mid = sum(
    float(x["mid"]) for x in (c_atm, p_atm)
    if x is not None and np.isfinite(x["mid"]) and x["mid"] > 0
)
implied_move = straddle_mid if (c_atm is not None and p_atm is not None) else float("nan")
implied_move_pct = implied_move / spot if np.isfinite(implied_move) else float("nan")

call_oi = float(calls["openInterest"].sum())
put_oi  = float(puts["openInterest"].sum())
pc_ratio = put_oi / call_oi if call_oi > 0 else float("nan")

ui.kpi_row([
    {"label": "Underlying", "value": f"${spot:,.2f}"},
    {"label": "Days to Expiry", "value": f"{dte}"},
    {"label": "ATM Implied Volatility", "value": f"{atm_iv:.1%}" if np.isfinite(atm_iv) else "N/A"},
    {"label": "Implied Move (straddle)",
     "value": f"±${implied_move:,.2f}" if np.isfinite(implied_move) else "N/A",
     "delta": f"±{implied_move_pct:.1%} of spot" if np.isfinite(implied_move_pct) else None,
     "delta_kind": "neu"},
    {"label": "Put/Call Open Interest", "value": f"{pc_ratio:.2f}" if np.isfinite(pc_ratio) else "N/A"},
])
if qtime is not None:
    ui.data_asof_caption(qtime, "yfinance")

# ── IV smile ──────────────────────────────────────────────────────────────────

with ui.panel(f"Implied Volatility Smile — {expiry}"):
    if calls_band.empty and puts_band.empty:
        ui.banner("info", "No usable strikes inside the selected band.")
    else:
        smile = go.Figure()
        if not calls_band.empty:
            smile.add_trace(go.Scatter(
                x=calls_band["strike"], y=calls_band["impliedVolatility"] * 100,
                mode="lines+markers", name="Calls",
                line=dict(color=PRIMARY, width=1.6), marker=dict(size=5),
            ))
        if not puts_band.empty:
            smile.add_trace(go.Scatter(
                x=puts_band["strike"], y=puts_band["impliedVolatility"] * 100,
                mode="lines+markers", name="Puts",
                line=dict(color=BENCHMARK, width=1.6), marker=dict(size=5),
            ))
        smile.add_vline(
            x=spot, line_dash="dot", line_color=NEUTRAL, line_width=1.2,
            annotation_text=f"Spot ${spot:,.2f}",
            annotation_position="top right", annotation_font_size=11,
        )
        smile.update_layout(
            xaxis_title="Strike ($)", yaxis_title="Implied volatility (%)",
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", legend=dict(x=0.02, y=0.98),
        )
        apply_chart_theme(smile)
        st.plotly_chart(smile, width="stretch", config=CHART_CONFIG)
        st.caption(
            "Chain-reported implied volatilities at the mid quote. Strikes with "
            "no open interest, missing IVs, or dead quotes are dropped rather "
            "than plotted."
        )

# ── Term structure ────────────────────────────────────────────────────────────

with ui.panel("ATM Implied Volatility Term Structure"):
    with st.spinner(f"Building term structure across {len(expiries)} expiries..."):
        term = _atm_term_structure(ticker, tuple(expiries), spot)

    if term.empty:
        ui.banner("info", "No expiries produced a usable ATM quote.")
    else:
        ts_fig = go.Figure(go.Scatter(
            x=pd.to_datetime(term["expiry"]), y=term["atm_iv"] * 100,
            mode="lines+markers", line=dict(color=PRIMARY, width=1.8),
            marker=dict(size=6),
            hovertemplate="%{x|%Y-%m-%d}<br>ATM IV: %{y:.1f}%<extra></extra>",
        ))
        sel_iv = term.loc[term["expiry"] == expiry, "atm_iv"]
        if len(sel_iv):
            ts_fig.add_trace(go.Scatter(
                x=[pd.Timestamp(expiry)], y=[float(sel_iv.iloc[0]) * 100],
                mode="markers", marker=dict(size=11, color=BENCHMARK, symbol="diamond"),
                name="Selected expiry", showlegend=False,
            ))
        ts_fig.update_layout(
            xaxis_title="Expiry", yaxis_title="ATM implied volatility (%)",
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", showlegend=False,
        )
        apply_chart_theme(ts_fig)
        st.plotly_chart(ts_fig, width="stretch", config=CHART_CONFIG)
        st.caption(
            "ATM straddle implied volatility per listed expiry. Building this "
            "curve makes one chain request per expiry; results are cached for "
            "15 minutes, so the first load of a new symbol is the slow one. "
            "An inverted front end is typical ahead of scheduled events."
        )

# ── Positioning ───────────────────────────────────────────────────────────────

with ui.panel(f"Open Interest and Volume by Strike — {expiry}"):
    metric = st.radio("Metric", ["Open Interest", "Volume"], horizontal=True,
                      label_visibility="collapsed")
    col = "openInterest" if metric == "Open Interest" else "volume"

    if calls_band.empty and puts_band.empty:
        ui.banner("info", "No usable strikes inside the selected band.")
    else:
        prof = go.Figure()
        if not calls_band.empty:
            prof.add_trace(go.Bar(
                y=calls_band["strike"], x=calls_band[col],
                orientation="h", name="Calls", marker_color=POSITIVE,
                hovertemplate="Strike %{y}<br>Calls: %{x:,.0f}<extra></extra>",
            ))
        if not puts_band.empty:
            prof.add_trace(go.Bar(
                y=puts_band["strike"], x=-puts_band[col],
                orientation="h", name="Puts", marker_color=NEGATIVE,
                hovertemplate="Strike %{y}<br>Puts: %{customdata:,.0f}<extra></extra>",
                customdata=puts_band[col],
            ))
        prof.add_hline(y=spot, line_dash="dot", line_color=NEUTRAL, line_width=1.2,
                       annotation_text=f"Spot ${spot:,.2f}",
                       annotation_position="top right", annotation_font_size=11)
        prof.add_vline(x=0, line_color=REFLINE, line_width=1)
        prof.update_layout(
            barmode="overlay",
            xaxis_title=f"{metric} (puts mirrored left, calls right)",
            yaxis_title="Strike ($)",
            height=max(360, 14 * max(len(calls_band), len(puts_band)) + 80),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.02, y=0.98),
        )
        apply_chart_theme(prof)
        st.plotly_chart(prof, width="stretch", config=CHART_CONFIG)

# ── Chain table ───────────────────────────────────────────────────────────────

def _table(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "Strike":        df["strike"],
        "Bid":           df["bid"],
        "Ask":           df["ask"],
        "Mid":           df["mid"],
        "IV":            df["impliedVolatility"],
        "Delta":         [op.bs_delta(opt_type, spot, k, T, r, iv)
                          for k, iv in zip(df["strike"], df["impliedVolatility"])],
        "Volume":        df["volume"].astype(int),
        "Open Interest": df["openInterest"].astype(int),
    })
    return out.reset_index(drop=True)


_CHAIN_CFG = {
    "Strike":        st.column_config.NumberColumn(format="%.2f"),
    "Bid":           st.column_config.NumberColumn(format="$%.2f"),
    "Ask":           st.column_config.NumberColumn(format="$%.2f"),
    "Mid":           st.column_config.NumberColumn(format="$%.2f"),
    "IV":            st.column_config.NumberColumn(format="%.1%"),
    "Delta":         st.column_config.NumberColumn(format="%.3f",
                         help="Black-Scholes-Merton delta at the chain-reported IV"),
    "Volume":        st.column_config.NumberColumn(format="%d"),
    "Open Interest": st.column_config.NumberColumn(format="%d"),
}

with ui.panel(f"Quote Table — {expiry}, strikes within ±{strike_band}% of spot"):
    side = st.radio("Side", ["Calls", "Puts"], horizontal=True,
                    label_visibility="collapsed", key="chain_side")
    band_df = calls_band if side == "Calls" else puts_band
    if band_df.empty:
        ui.banner("info", f"No usable {side.lower()} inside the selected band.")
    else:
        tbl = _table(band_df, side[:-1].lower())
        st.dataframe(tbl, column_config=_CHAIN_CFG, hide_index=True,
                     width="stretch", height=min(520, 38 * len(tbl) + 60))
        ui.download_row(tbl, f"{ticker}_{expiry}_{side.lower()}")

st.caption(
    "Quotes are delayed and sourced from Yahoo Finance; bids, asks, and "
    "implied volatilities are indicative only and may be stale for illiquid "
    "strikes. Deltas are model values, not exchange-reported Greeks."
)

ui.footer_disclaimer()
