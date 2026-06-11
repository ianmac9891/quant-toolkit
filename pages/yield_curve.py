"""Yield Curve Monitor — Treasury curve level, history, and inversion tracking.

Data are the CBOE Treasury yield indices on Yahoo: ^IRX (13-week bill),
^FVX (5-year), ^TNX (10-year), ^TYX (30-year).

Unit note: the CBOE convention quotes these indices in yield points equal to
ten times the percentage yield (42.5 means 4.25%). The currently installed
yfinance, however, returns them already normalized to percent (4.25 means
4.25%), verified against live quotes at build time. _normalize() therefore
divides by ten only when a series' recent level is implausibly high for a
Treasury yield (above 25), which handles either convention without trusting
the provider to stay consistent.

This page also exports the latest 13-week bill rate as the app-wide default
risk-free rate: the Derivatives Workbench and Options Chain Explorer rate
inputs pre-fill from it once this page has been visited in the session.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.theme import (
    PRIMARY, BENCHMARK, POSITIVE, NEGATIVE, NEUTRAL, REFLINE, TEXT,
    NEGATIVE_18, CHART_CONFIG, apply_chart_theme,
)

ui.page_header(
    "Macro & Rates", "Yield Curve Monitor",
    "The Treasury curve from the 13-week bill to the 30-year bond: current "
    "shape, history by tenor, and the spreads whose inversions have preceded "
    "recessions. The current bill rate becomes the app-wide risk-free default.",
)

_TENORS = {
    "^IRX": ("13-Week Bill", 0.25),
    "^FVX": ("5-Year Note", 5.0),
    "^TNX": ("10-Year Note", 10.0),
    "^TYX": ("30-Year Bond", 30.0),
}


def _normalize(series: pd.Series) -> pd.Series:
    """Percent yields regardless of the provider's quoting convention."""
    s = series.dropna()
    if s.empty:
        return s
    return s / 10.0 if float(s.tail(20).median()) > 25.0 else s


# ── Parameters ────────────────────────────────────────────────────────────────

with ui.panel("Parameters"):
    lookback_label = st.select_slider(
        "History Lookback",
        options=["1 year", "3 years", "5 years", "10 years"],
        value="5 years",
    )
lookback_years = int(lookback_label.split()[0])

today = date.today()
start = today - timedelta(days=365 * lookback_years + 10)

# ── Data ──────────────────────────────────────────────────────────────────────

with st.spinner("Retrieving Treasury yield indices..."):
    frames = ui.fetch_universe(tuple(_TENORS), start, today)

yields = pd.DataFrame({
    t: _normalize(df["close"])
    for t, df in frames.items()
    if not df.empty and "close" in df.columns
}).sort_index().ffill().dropna(how="all")

missing = sorted(set(_TENORS) - set(yields.columns))
if yields.empty or len(yields.columns) < 2:
    ui.data_unavailable("Treasury yield indices")
    st.stop()
if missing:
    ui.banner("warn", f"No data for: <span class='mono'>{', '.join(missing)}</span> — "
                      "curve shown without those tenors.")

ui.data_asof_caption(yields.index.max())

latest = yields.iloc[-1]
year_ago_idx = yields.index[yields.index <= yields.index.max() - pd.Timedelta(days=365)]
year_ago = yields.loc[year_ago_idx[-1]] if len(year_ago_idx) else None

# Publish the app-wide risk-free default from the bill rate
if "^IRX" in latest.index and np.isfinite(latest["^IRX"]):
    ui.set_default_rf_pct(float(latest["^IRX"]))

# ── Headline ──────────────────────────────────────────────────────────────────

kpis = []
for t, (name, _) in _TENORS.items():
    if t in latest.index and np.isfinite(latest[t]):
        delta = None
        if year_ago is not None and t in year_ago.index and np.isfinite(year_ago[t]):
            delta = f"{(latest[t] - year_ago[t]) * 100:+.0f} bp vs 1y ago"
        kpis.append({"label": name, "value": f"{latest[t]:.2f}%",
                     "delta": delta, "delta_kind": "neu"})
if {"^TNX", "^IRX"}.issubset(latest.index):
    spread_10y_13w = latest["^TNX"] - latest["^IRX"]
    kpis.append({
        "label": "10-Year minus 13-Week",
        "value": f"{spread_10y_13w * 100:+.0f} bp",
        "delta": "inverted" if spread_10y_13w < 0 else "upward",
        "delta_kind": "neg" if spread_10y_13w < 0 else "pos",
    })
ui.kpi_row(kpis)

if "^IRX" in latest.index and np.isfinite(latest["^IRX"]):
    ui.banner(
        "info",
        f"The current 13-week bill rate of <b>{latest['^IRX']:.2f}%</b> is now "
        "the default risk-free rate across the terminal's rate inputs "
        "(Derivatives Workbench, Options Chain Explorer). Each input remains "
        "editable per analysis.",
    )

# ── Current curve ─────────────────────────────────────────────────────────────

with ui.panel("Current Curve"):
    mats = [_TENORS[t][1] for t in yields.columns]
    order = np.argsort(mats)
    cols_sorted = [yields.columns[i] for i in order]
    mats_sorted = [_TENORS[t][1] for t in cols_sorted]

    curve = go.Figure()
    curve.add_trace(go.Scatter(
        x=mats_sorted, y=[latest[t] for t in cols_sorted],
        mode="lines+markers+text",
        line=dict(color=PRIMARY, width=2), marker=dict(size=9),
        text=[f"{latest[t]:.2f}%" for t in cols_sorted],
        textposition="top center",
        name=f"Latest ({yields.index.max().date()})",
        hovertemplate="%{x}y: %{y:.2f}%<extra></extra>",
    ))
    if year_ago is not None:
        curve.add_trace(go.Scatter(
            x=mats_sorted, y=[year_ago[t] for t in cols_sorted],
            mode="lines+markers",
            line=dict(color=NEUTRAL, width=1.4, dash="dash"), marker=dict(size=6),
            name="One year ago",
            hovertemplate="%{x}y: %{y:.2f}%<extra></extra>",
        ))
    curve.update_layout(
        xaxis_title="Maturity (years)", yaxis_title="Yield (%)",
        xaxis=dict(tickvals=mats_sorted,
                   ticktext=[f"{m:g}y" if m >= 1 else "13w" for m in mats_sorted]),
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(curve)
    st.plotly_chart(curve, width="stretch", config=CHART_CONFIG)

# ── History by tenor ──────────────────────────────────────────────────────────

_SERIES_COLORS = {"^IRX": NEUTRAL, "^FVX": BENCHMARK, "^TNX": PRIMARY, "^TYX": TEXT}

with ui.panel(f"Yield History — {lookback_label}"):
    hist = go.Figure()
    for t in yields.columns:
        hist.add_trace(go.Scatter(
            x=yields.index, y=yields[t], mode="lines",
            name=_TENORS[t][0],
            line=dict(color=_SERIES_COLORS.get(t, PRIMARY), width=1.4),
        ))
    hist.update_layout(
        yaxis_title="Yield (%)", height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", legend=dict(x=0.02, y=0.98),
    )
    apply_chart_theme(hist)
    st.plotly_chart(hist, width="stretch", config=CHART_CONFIG)

# ── Spread tracker ────────────────────────────────────────────────────────────

with ui.panel("Curve Spreads and Inversions"):
    spreads = pd.DataFrame(index=yields.index)
    if {"^TNX", "^IRX"}.issubset(yields.columns):
        spreads["10y minus 13w"] = (yields["^TNX"] - yields["^IRX"]) * 100
    if {"^TYX", "^FVX"}.issubset(yields.columns):
        spreads["30y minus 5y"] = (yields["^TYX"] - yields["^FVX"]) * 100
    spreads = spreads.dropna(how="all")

    if spreads.empty:
        ui.banner("info", "Both legs of each spread are required; tenors are missing.")
    else:
        sp = go.Figure()
        # Shade inversion: the portion of the primary spread below zero
        primary_col = spreads.columns[0]
        inverted = spreads[primary_col].clip(upper=0)
        sp.add_trace(go.Scatter(
            x=spreads.index, y=inverted, mode="lines",
            line=dict(width=0), fill="tozeroy", fillcolor=NEGATIVE_18,
            name="Inversion (10y under 13w)", hoverinfo="skip", showlegend=True,
        ))
        colors = [PRIMARY, BENCHMARK]
        for i, col in enumerate(spreads.columns):
            sp.add_trace(go.Scatter(
                x=spreads.index, y=spreads[col], mode="lines",
                name=col, line=dict(color=colors[i % 2], width=1.5),
            ))
        sp.add_hline(y=0, line_color=REFLINE, line_width=1.2)
        sp.update_layout(
            yaxis_title="Spread (basis points)", height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", legend=dict(x=0.02, y=0.98),
        )
        apply_chart_theme(sp)
        st.plotly_chart(sp, width="stretch", config=CHART_CONFIG)
        st.caption(
            "Shaded regions mark periods where the 10-year yield sat below the "
            "13-week bill rate. Curve inversions have preceded United States "
            "recessions with long and variable lags; an inversion is a statement "
            "about expected policy rates, not a timing signal."
        )

        latest_spreads = spreads.iloc[-1]
        st.dataframe(
            pd.DataFrame({
                "Current (bp)": latest_spreads.round(0).astype(int),
                "1y Range (bp)": [
                    f"{spreads[c].tail(252).min():.0f} to {spreads[c].tail(252).max():.0f}"
                    for c in spreads.columns
                ],
            }),
            width="stretch",
        )
        ui.download_row(spreads.round(1), "curve_spreads")

ui.footer_disclaimer()
