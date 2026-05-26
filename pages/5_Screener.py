"""Bullish stock screener — medium-horizon momentum and trend ranking."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import screener as sc
from src.theme import NEUTRAL, apply_chart_theme

st.set_page_config(page_title="Screener", layout="wide")

# ── Sidebar: universe ─────────────────────────────────────────────────────────

st.sidebar.header("Universe")

universe_mode = st.sidebar.radio(
    "Universe",
    [
        "Custom watchlist",
        "S&P 500 (~500 tickers)",
        "S&P 1500 (~1500 tickers, longer scan)",
    ],
    label_visibility="collapsed",
)

if universe_mode == "Custom watchlist":
    raw = st.sidebar.text_area(
        "Tickers",
        "AAPL\nMSFT\nNVDA\nGOOGL\nAMZN\nMETA\nTSLA\nJPM\nV\nUNH",
        height=160,
        help="Enter one ticker per line or separate them with commas.",
    )
    tickers_input = sorted(set(t for t in re.split(r"[\s,]+", raw.strip().upper()) if t))
else:
    tickers_input = None

# ── Sidebar: date range + filters ─────────────────────────────────────────────

st.sidebar.header("Fetch window")
today = date.today()

start_date = st.sidebar.date_input(
    "Start date",
    value=today - timedelta(days=730),
    min_value=date(2000, 1, 1),
    max_value=today - timedelta(days=280),
    help=(
        "Signals are computed over the trailing 252 trading days. "
        "A longer fetch window ensures that tickers with partial coverage "
        "still meet the minimum observation threshold."
    ),
)
end_date = today

st.sidebar.header("Filters")

MCAP_OPTIONS = {
    "No filter":  0,
    "> $500 M":     500_000_000,
    "> $1 B":     1_000_000_000,
    "> $5 B":     5_000_000_000,
    "> $10 B":   10_000_000_000,
}
min_mcap_label = st.sidebar.selectbox("Min market cap", list(MCAP_OPTIONS))
min_market_cap = MCAP_OPTIONS[min_mcap_label]

max_extension = st.sidebar.slider(
    "Exclude stocks extended > X σ above trendline",
    min_value=1.0, max_value=5.0, value=5.0, step=0.5,
    help=(
        "Extension σ measures how many standard deviations the latest price sits "
        "above its own 252-day OLS trendline. At 5.0 the filter is inactive. "
        "Setting this to 2.0 removes names flagged as 'Stretched'."
    ),
)

# ── Cached helpers (fast paths) ───────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _sp500_tickers() -> list[str]:
    return sc.fetch_sp500_tickers()

@st.cache_data(ttl=86400, show_spinner=False)
def _sp1500_tickers() -> list[str]:
    return sc.fetch_sp1500_tickers()

@st.cache_data(ttl=86400, show_spinner=False)
def _market_caps(tickers: tuple[str, ...]) -> pd.Series:
    return sc.fetch_market_caps(list(tickers))

@st.cache_data(ttl=86400, show_spinner=False)
def _rev_growth(tickers: tuple[str, ...]) -> pd.Series:
    return sc.fetch_rev_growth(list(tickers))

@st.cache_data(ttl=3600, show_spinner=False)
def _screen(
    prices: pd.DataFrame,
    market_caps: pd.Series,
    rev_growth: pd.Series,
    min_mcap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return sc.run_screen(prices, market_caps, rev_growth, min_mcap)


# ── Resolve universe ──────────────────────────────────────────────────────────

if universe_mode == "S&P 500 (~500 tickers)":
    with st.spinner("Loading S&P 500 constituents..."):
        tickers_input = _sp500_tickers()
elif universe_mode == "S&P 1500 (~1500 tickers, longer scan)":
    with st.spinner("Loading S&P 1500 constituents..."):
        tickers_input = _sp1500_tickers()

if not tickers_input:
    st.warning("Enter at least one ticker.")
    st.stop()

tickers_key = tuple(sorted(tickers_input))
n_requested = len(tickers_key)

# ── Price fetch with per-ticker retry + progress bar ─────────────────────────

prices_cache_key = (tickers_key, str(start_date), str(end_date))
needs_fetch = st.session_state.get("_prices_key") != prices_cache_key

if needs_fetch:
    n = n_requested
    est_minutes = max(1, round(n * 0.25 / 60, 1))
    st.info(
        f"First scan: downloading prices for {n} tickers one-by-one via the "
        f"parquet cache (~{est_minutes} min). Re-runs in this session are instant; "
        "re-runs in a new session read from local cache and take ~10 s."
    )

    prog = st.progress(0.0, text=f"Fetching prices: 0 / {n}")
    price_dict: dict[str, pd.Series] = {}
    dl_failed:  list[str]            = []
    done = 0

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(sc.fetch_ticker_prices, t, start_date, end_date, 3): t
            for t in tickers_key
        }
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            prog.progress(done / n, text=f"Fetching prices: {done} / {n}  ({t})")
            try:
                series = fut.result()
                if not series.empty:
                    price_dict[t] = series
                else:
                    dl_failed.append(t)
            except Exception:
                dl_failed.append(t)

    prog.empty()

    price_df = pd.DataFrame(price_dict).sort_index() if price_dict else pd.DataFrame()

    st.session_state["_prices_key"] = prices_cache_key
    st.session_state["_price_df"]   = price_df
    st.session_state["_dl_failed"]  = dl_failed

else:
    price_df = st.session_state["_price_df"]
    dl_failed = st.session_state["_dl_failed"]

if price_df.empty:
    st.error("No price data returned. Check tickers and date range.")
    st.stop()

if len(price_df) < 252:
    st.warning(
        f"Only {len(price_df)} trading days in the selected range — "
        "need ≥ 252 for all signals. Extend the start date."
    )
    st.stop()

active_tickers = tuple(sorted(price_df.columns.tolist()))

# ── Market caps (gates the filter) ───────────────────────────────────────────

with st.spinner(f"Fetching market caps for {len(active_tickers)} tickers..."):
    mcaps = _market_caps(active_tickers)

# ── Rev growth (best-effort, 60 s cap) ───────────────────────────────────────

with st.spinner("Fetching revenue growth (best-effort, up to 60 s)..."):
    rev_growth = _rev_growth(active_tickers)

# ── Run screen ────────────────────────────────────────────────────────────────

with st.spinner("Computing signals and scores..."):
    ranked, insufficient, suspect = _screen(price_df, mcaps, rev_growth, float(min_market_cap))

# Extension filter (no re-download; pure client-side slice)
if max_extension < 5.0:
    mask = ranked["extension_z"].isna() | (ranked["extension_z"] <= max_extension)
    ranked = ranked[mask]

# ── Header metrics ────────────────────────────────────────────────────────────

st.title("Screener results")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requested",            n_requested)
c2.metric("Ranked",               len(ranked))
c3.metric("Insufficient history", len(insufficient))
c4.metric("Suspect data",         len(suspect))
c5.metric("Download failed",      len(dl_failed))

# ── Section 1: Ranked table ───────────────────────────────────────────────────

st.header("Ranked tickers")

if ranked.empty:
    st.info("No tickers passed all filters with sufficient history.")
else:
    display_cols = [
        "composite",
        "extension_z", "extension_flag",
        "mom_12_1", "mom_6m",
        "pct_above_200sma", "golden_cross", "dist_52w_high",
        "trend_slope", "trend_r2",
        "trailing_vol", "market_cap", "rev_growth_yoy",
    ]
    disp = ranked[[c for c in display_cols if c in ranked.columns]].copy()
    disp.index.name = "ticker"

    for col in ("mom_12_1", "mom_6m", "trailing_vol", "trend_slope", "rev_growth_yoy"):
        if col in disp.columns:
            disp[col] = disp[col] * 100

    disp["market_cap_b"] = disp["market_cap"] / 1e9
    if "golden_cross" in disp.columns:
        disp["golden_cross"] = disp["golden_cross"].astype("boolean")

    col_cfg = {
        "composite":        st.column_config.NumberColumn("Composite",          format="%.2f"),
        "extension_z":      st.column_config.NumberColumn("Extension σ",        format="%.2f",
                                help="Std devs above own 252-day trendline"),
        "extension_flag":   st.column_config.TextColumn("Extension",
                                help="On trend / Extended / Stretched"),
        "mom_12_1":         st.column_config.NumberColumn("12-1 Mom",           format="%.1f%%"),
        "mom_6m":           st.column_config.NumberColumn("6m Mom",             format="%.1f%%"),
        "pct_above_200sma": st.column_config.NumberColumn("% vs SMA200",        format="%.1f%%"),
        "golden_cross":     st.column_config.CheckboxColumn("GX",
                                help="SMA50 > SMA200"),
        "dist_52w_high":    st.column_config.NumberColumn("vs 52w High",        format="%.1f%%"),
        "trend_slope":      st.column_config.NumberColumn("Trend Slope (ann)",  format="%.1f%%"),
        "trend_r2":         st.column_config.NumberColumn("Trend R²",           format="%.2f"),
        "trailing_vol":     st.column_config.NumberColumn("Realized Vol",       format="%.1f%%"),
        "market_cap_b":     st.column_config.NumberColumn("Mkt Cap ($B)",       format="$%.1f"),
        "rev_growth_yoy":   st.column_config.NumberColumn("Rev Growth YoY",     format="%.1f%%"),
    }

    st.dataframe(
        disp.drop(columns=["market_cap"], errors="ignore"),
        column_config=col_cfg,
        use_container_width=True,
        height=520,
    )

    csv = ranked[display_cols].reset_index().to_csv(index=False)
    st.download_button(
        "Download full results (CSV)",
        csv,
        file_name="screener_results.csv",
        mime="text/csv",
    )

# ── Section 2: Scatter plot ───────────────────────────────────────────────────

st.header("Momentum vs trend quality")

if not ranked.empty:
    plot_df = ranked[
        ["mom_12_1", "trend_r2", "composite", "extension_z", "extension_flag"]
    ].dropna(subset=["mom_12_1", "trend_r2", "composite"])

    if not plot_df.empty:
        c_min   = float(plot_df["composite"].min())
        c_max   = float(plot_df["composite"].max())
        c_range = max(c_max - c_min, 1e-6)
        sizes   = ((plot_df["composite"] - c_min) / c_range * 16 + 6).tolist()

        scatter_fig = go.Figure(go.Scatter(
            x=(plot_df["mom_12_1"] * 100).tolist(),
            y=plot_df["trend_r2"].tolist(),
            mode="markers+text",
            text=[""] * len(plot_df),
            textposition="top center",
            textfont=dict(size=9),
            customdata=list(zip(
                plot_df.index.tolist(),
                plot_df["composite"].round(2).tolist(),
                [f"{v:.2f}" if pd.notna(v) else "—" for v in plot_df["extension_z"]],
                plot_df["extension_flag"].fillna("").tolist(),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Composite: %{customdata[1]}<br>"
                "12-1 Mom: %{x:.1f}%<br>"
                "Trend R²: %{y:.2f}<br>"
                "Extension: %{customdata[2]}σ (%{customdata[3]})"
                "<extra></extra>"
            ),
            marker=dict(
                size=sizes,
                color=plot_df["composite"].tolist(),
                colorscale="Plasma",
                cmin=c_min, cmax=c_max,
                colorbar=dict(title="Composite", thickness=14),
                line=dict(width=0.5, color="rgba(229,231,235,0.5)"),
            ),
        ))

        top20 = set(plot_df.nlargest(20, "composite").index)
        scatter_fig.data[0].text = [
            t if t in top20 else "" for t in plot_df.index
        ]

        scatter_fig.add_annotation(
            x=0.97, y=0.97, xref="paper", yref="paper",
            text="Strong & clean uptrend",
            showarrow=False,
            font=dict(size=11, color="rgba(229,231,235,0.8)"),
            bgcolor="rgba(20,20,30,0.7)",
            borderpad=4,
        )
        scatter_fig.update_layout(
            xaxis_title="12-1 Momentum (%)",
            yaxis_title="Trend R²",
            yaxis=dict(range=[-0.05, 1.05]),
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="closest",
        )
        apply_chart_theme(scatter_fig)
        st.plotly_chart(scatter_fig, use_container_width=True)
        st.caption(
            "Color and size encode composite score (brighter / larger = higher rank). "
            "Top-right combines strong trailing momentum with a smooth, persistent trend. "
            "Hover to see extension σ — names far above their trendline may be stretched entries."
        )

# ── Expanders for excluded tickers ────────────────────────────────────────────

if not insufficient.empty:
    with st.expander(
        f"Insufficient price history — {len(insufficient)} ticker{'s' if len(insufficient) != 1 else ''}"
    ):
        st.info(
            "These tickers returned price data but had fewer than 80% of the required "
            "252 trading days in the selected date range (recent IPOs, etc.)."
        )
        st.dataframe(
            pd.DataFrame({"ticker": insufficient.index}).reset_index(drop=True),
            use_container_width=True,
        )

if not suspect.empty:
    with st.expander(
        f"Suspect data — {len(suspect)} ticker{'s' if len(suspect) != 1 else ''}"
    ):
        st.warning(
            "These tickers had non-physical signal values (12-1 momentum > 500% or "
            "6-month momentum > 300%), almost certainly caused by spinoff stub prices, "
            "unadjusted splits, or a reused ticker symbol. They were excluded before "
            "z-scoring so they don't inflate the cross-sectional standard deviation and "
            "distort every other stock's composite score."
        )
        disp_suspect = suspect[["mom_12_1", "mom_6m"]].copy()
        disp_suspect["mom_12_1"] = disp_suspect["mom_12_1"] * 100
        disp_suspect["mom_6m"]   = disp_suspect["mom_6m"] * 100
        disp_suspect = disp_suspect.sort_values("mom_12_1", key=abs, ascending=False)
        disp_suspect.index.name = "ticker"
        st.dataframe(
            disp_suspect,
            column_config={
                "mom_12_1": st.column_config.NumberColumn("12-1 Mom", format="%.1f%%"),
                "mom_6m":   st.column_config.NumberColumn("6m Mom",   format="%.1f%%"),
            },
            use_container_width=True,
        )

if dl_failed:
    with st.expander(
        f"Download failed — {len(dl_failed)} ticker{'s' if len(dl_failed) != 1 else ''}"
    ):
        st.warning(
            "These tickers returned no data after 3 retries. "
            "They may be delisted, use a different symbol, or had persistent network errors."
        )
        st.dataframe(
            pd.DataFrame({"ticker": sorted(dl_failed)}).reset_index(drop=True),
            use_container_width=True,
        )

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.markdown(
    "_This screener ranks stocks by characteristics historically associated with "
    "medium-term trend continuation. It is a research funnel, not a buy list — "
    "past trend does not guarantee future returns._"
)
