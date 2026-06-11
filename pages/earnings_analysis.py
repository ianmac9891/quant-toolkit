"""Earnings Move Analysis — realized earnings-day reactions vs the option-implied move.

Yahoo reports announcement timestamps with limited precision (most rows carry
a 16:00 ET placeholder regardless of actual timing), so rather than guessing
pre-market vs post-market per event, every move here uses one stated
convention: the close-to-next-close return from the last session at or before
the announcement date. For after-close reporters this is exactly the reaction
session; for pre-market reporters it lags by one session. The caption states
the convention so the numbers can be read for what they are.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import ui
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE,
    CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Equity & Derivatives Research", "Earnings Move Analysis",
    "Historical earnings-day price reactions over a configurable lookback, "
    "with the current option-implied move alongside the realized distribution "
    "when a post-earnings expiry is listed. A comparison of implied versus "
    "trailing realized moves, not a prediction.",
)

# ── Cached access ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def _earnings_dates(ticker: str, limit: int) -> pd.DataFrame:
    """Announcement dates from Yahoo. Returns an empty frame when the
    provider has nothing (ETFs, funds, many foreign listings)."""
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def _implied_move(ticker: str, after: date) -> tuple:
    """(expiry, implied move fraction) from the ATM straddle of the first
    listed expiry at or after `after`, or (None, nan). One chain request."""
    try:
        tk = yf.Ticker(ticker)
        expiries = [e for e in tk.options
                    if after <= pd.Timestamp(e).date() <= after + timedelta(days=14)]
        if not expiries:
            return None, float("nan")
        expiry = expiries[0]
        ch = tk.option_chain(expiry)
        spot = None
        if isinstance(getattr(ch, "underlying", None), dict):
            spot = ch.underlying.get("regularMarketPrice")
        if not spot or spot <= 0:
            return None, float("nan")

        def _atm_mid(df):
            df = df[(df["openInterest"].fillna(0) > 0)
                    & ~((df["bid"].fillna(0) <= 0) & (df["ask"].fillna(0) <= 0))]
            if df.empty:
                return float("nan")
            row = df.loc[(df["strike"] - spot).abs().idxmin()]
            bid, ask = float(row["bid"] or 0), float(row["ask"] or 0)
            return (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)

        straddle = _atm_mid(ch.calls) + _atm_mid(ch.puts)
        if not np.isfinite(straddle) or straddle <= 0:
            return None, float("nan")
        return expiry, float(straddle / spot)
    except Exception:
        return None, float("nan")


# ── Parameters ────────────────────────────────────────────────────────────────

with ui.panel("Parameters"):
    c1, c2 = st.columns([1, 1.6])
    with c1:
        ticker = st.text_input("Instrument", value=ui.get_default_ticker("AAPL")).upper().strip()
    with c2:
        lookback = st.slider(
            "Lookback (reported quarters)", min_value=4, max_value=24, value=12,
            help="Number of past announcements included in the realized "
                 "distribution, subject to provider history.",
        )

if not ticker:
    ui.banner("info", "Enter an instrument symbol to begin.")
    st.stop()

# ── Earnings dates ────────────────────────────────────────────────────────────

with st.spinner(f"Retrieving {ticker} earnings calendar..."):
    # Generous limit: the frame includes upcoming dates and occasional gaps
    earnings = _earnings_dates(ticker, limit=lookback + 8)

if earnings.empty:
    ui.data_unavailable(f"{ticker}: no earnings dates returned; symbol may not "
                        "report earnings (funds, indices) or history is missing")
    st.stop()

idx = pd.DatetimeIndex(earnings.index)
if idx.tz is not None:
    idx = idx.tz_localize(None)
ann_dates = pd.Series(idx.normalize(), index=earnings.index)

today = pd.Timestamp(date.today())
past_dates = sorted({d for d in ann_dates if d < today})[-lookback:]
future_dates = sorted({d for d in ann_dates if d >= today})
next_earnings = future_dates[0] if future_dates else None

if not past_dates:
    ui.data_unavailable(f"{ticker}: no past announcements within provider history")
    st.stop()

# ── Prices and realized moves ─────────────────────────────────────────────────

px_start = (past_dates[0] - pd.Timedelta(days=15)).date()
with st.spinner(f"Retrieving {ticker} price history..."):
    result = ui.fetch_prices(ticker, px_start, date.today())

if not result.ok or "adj_close" not in result.df.columns:
    ui.data_unavailable(f"{ticker}: {result.error or 'no usable columns'}")
    st.stop()

ui.remember_ticker(ticker)
closes = result.df["adj_close"].dropna()
ui.data_asof_caption(result.asof, result.source)

moves = []
for d in past_dates:
    on_or_before = closes.index[closes.index <= d]
    if len(on_or_before) == 0:
        continue
    t0 = on_or_before[-1]
    after = closes.index[closes.index > t0]
    if len(after) == 0:
        continue  # announcement too recent for a next close
    t1 = after[0]
    moves.append({
        "announcement": d.date(),
        "from": t0.date(),
        "to": t1.date(),
        "return": float(closes.loc[t1] / closes.loc[t0] - 1.0),
    })

if not moves:
    ui.data_unavailable(f"{ticker}: price history does not cover the announcement dates")
    st.stop()

mv = pd.DataFrame(moves).set_index("announcement").sort_index()
abs_moves = mv["return"].abs()

# ── Implied move for the next announcement ────────────────────────────────────

implied_expiry, implied_frac = (None, float("nan"))
if next_earnings is not None:
    with st.spinner("Checking listed expiries after the next announcement..."):
        implied_expiry, implied_frac = _implied_move(ticker, next_earnings.date())

# ── Headline ──────────────────────────────────────────────────────────────────

kpis = [
    {"label": "Announcements Analyzed", "value": f"{len(mv)}"},
    {"label": "Mean Absolute Move", "value": f"{abs_moves.mean() * 100:.1f}%"},
    {"label": "Median Absolute Move", "value": f"{abs_moves.median() * 100:.1f}%"},
    {"label": "Largest Move", "value": f"{abs_moves.max() * 100:.1f}%"},
    {"label": "Positive Reactions", "value": f"{(mv['return'] > 0).mean() * 100:.0f}%"},
]
if next_earnings is not None:
    kpis.append({
        "label": "Next Announcement",
        "value": str(next_earnings.date()),
        "delta": (f"implied move ±{implied_frac:.1%}"
                  if np.isfinite(implied_frac) else "no usable post-event expiry"),
        "delta_kind": "neu",
    })
ui.kpi_row(kpis)

if np.isfinite(implied_frac) and implied_expiry is not None:
    ui.banner(
        "info",
        f"The {implied_expiry} ATM straddle currently prices a move of about "
        f"<b>±{implied_frac:.1%}</b> through the {next_earnings.date()} "
        f"announcement. The trailing {len(mv)}-quarter median realized move is "
        f"<b>{abs_moves.median():.1%}</b>. This is a comparison of implied "
        "versus trailing realized moves; it carries no directional or "
        "predictive claim.",
    )

# ── Signed reactions ──────────────────────────────────────────────────────────

with ui.panel("Earnings-Day Returns by Announcement"):
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in mv["return"]]
    bars = go.Figure(go.Bar(
        x=[str(d) for d in mv.index], y=mv["return"] * 100,
        marker_color=colors,
        text=[f"{v * 100:+.1f}%" for v in mv["return"]],
        textposition="outside",
        customdata=np.stack([mv["from"].astype(str), mv["to"].astype(str)], axis=1),
        hovertemplate=("Announced %{x}<br>Close %{customdata[0]} to "
                       "%{customdata[1]}: %{y:+.2f}%<extra></extra>"),
    ))
    bars.add_hline(y=0, line_color=REFLINE, line_width=1)
    bars.update_layout(
        xaxis_title="Announcement date", yaxis_title="One-day gap return (%)",
        height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    apply_chart_theme(bars)
    st.plotly_chart(bars, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Convention: the move is the close-to-close return from the last "
        "session at or before the announcement date to the following session. "
        "Provider timestamps are not precise enough to distinguish pre-market "
        "from post-market reporting per event, so a single convention is "
        "applied throughout; for pre-market reporters the stated date lags the "
        "reaction session by one day."
    )

# ── Distribution ──────────────────────────────────────────────────────────────

with ui.panel("Distribution of Absolute Moves"):
    hist = go.Figure(go.Histogram(
        x=abs_moves * 100,
        nbinsx=max(8, min(20, len(mv))),
        marker_color=PRIMARY, opacity=0.8,
        hovertemplate="|move| %{x}%<br>count %{y}<extra></extra>",
    ))
    hist.add_vline(
        x=float(abs_moves.median() * 100), line_dash="dash",
        line_color=NEUTRAL, line_width=1.2,
        annotation_text=f"Median {abs_moves.median():.1%}",
        annotation_position="top right", annotation_font_size=11,
    )
    if np.isfinite(implied_frac):
        hist.add_vline(
            x=float(implied_frac * 100), line_dash="dot",
            line_color=BENCHMARK, line_width=1.6,
            annotation_text=f"Implied ±{implied_frac:.1%}",
            annotation_position="top left", annotation_font_size=11,
        )
    hist.update_layout(
        xaxis_title="Absolute one-day gap return (%)", yaxis_title="Count",
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    apply_chart_theme(hist)
    st.plotly_chart(hist, width="stretch", config=CHART_CONFIG)
    if np.isfinite(implied_frac):
        st.caption(
            "The dotted reference is the current straddle-implied move for the "
            "first listed expiry after the next announcement. Implied moves "
            "embed a volatility risk premium and have historically exceeded "
            "realized moves more often than not; the comparison is contextual, "
            "not a trading signal."
        )

# ── Detail table ──────────────────────────────────────────────────────────────

with ui.panel("Announcement Detail"):
    detail = mv.copy()
    detail["return"] = detail["return"] * 100
    detail.index.name = "Announcement"
    st.dataframe(
        detail.rename(columns={"from": "From Session", "to": "To Session",
                               "return": "Return (%)"}),
        column_config={
            "Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        },
        width="stretch",
    )
    ui.download_row(mv, f"{ticker}_earnings_moves")

ui.footer_disclaimer()
