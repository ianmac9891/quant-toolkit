"""Equity Screening — cross-sectional momentum and trend-quality ranking."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src import data
from src import screener as sc
from src.theme import CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Systematic Research", "Equity Screening",
    "Cross-sectional ranking of equities by medium-horizon momentum and trend "
    "quality. A quantitative research funnel — rankings reflect statistical "
    "patterns in historical prices, not recommendations.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

MCAP_OPTIONS = {
    "No constraint": 0,
    "Above $500M":   500_000_000,
    "Above $1B":     1_000_000_000,
    "Above $5B":     5_000_000_000,
    "Above $10B":    10_000_000_000,
}

with st.form("screen_params"):
    st.markdown('<p class="qrt-kicker">Screening Parameters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        universe_mode = st.radio(
            "Universe",
            ["Custom watchlist", "S&P 500 (approx. 500 names)",
             "S&P 1500 (approx. 1500 names; extended scan)"],
        )
        raw_watchlist = st.text_area(
            "Watchlist (used when Custom is selected)",
            "AAPL\nMSFT\nNVDA\nGOOGL\nAMZN\nMETA\nTSLA\nJPM\nV\nUNH",
            height=150,
            help="One symbol per line or comma-separated.",
        )
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            start_date = st.date_input(
                "History Window Start",
                value=today - timedelta(days=730),
                min_value=date(2000, 1, 1),
                max_value=today - timedelta(days=280),
                help="Signals use the trailing 252 sessions; a longer window lets "
                     "names with partial coverage clear the minimum-history threshold.",
            )
            min_mcap_label = st.selectbox("Market Capitalization Floor", list(MCAP_OPTIONS))
        with cc2:
            max_extension = st.slider(
                "Trendline Extension Ceiling (standard deviations)",
                min_value=1.0, max_value=5.0, value=5.0, step=0.5,
                help="Excludes names whose latest price sits more than this many "
                     "standard deviations above their own 252-session regression "
                     "trendline. At 5.0 the constraint is inactive.",
            )
    submitted = st.form_submit_button("Run Screen", type="primary")

end_date = today
min_market_cap = MCAP_OPTIONS[min_mcap_label]

import re
tickers_input = (
    sorted(set(t for t in re.split(r"[\s,]+", raw_watchlist.strip().upper()) if t))
    if universe_mode == "Custom watchlist" else None
)

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _sp500_constituents() -> pd.Series:
    return sc.fetch_sp500_constituents()

@st.cache_data(ttl=86400, show_spinner=False)
def _sp1500_constituents() -> pd.Series:
    return sc.fetch_sp1500_constituents()

@st.cache_data(ttl=86400, show_spinner=False)
def _market_caps(tickers: tuple[str, ...]) -> pd.Series:
    return sc.fetch_market_caps(list(tickers))

@st.cache_data(ttl=86400, show_spinner=False)
def _rev_growth(tickers: tuple[str, ...]) -> pd.Series:
    return sc.fetch_rev_growth(list(tickers))

@st.cache_data(ttl=3600, show_spinner=False)
def _screen(prices: pd.DataFrame, market_caps: pd.Series, rev_growth: pd.Series,
            min_mcap: float):
    return sc.run_screen(prices, market_caps, rev_growth, min_mcap)


# ── Resolve universe and sector map ───────────────────────────────────────────

sector_map = pd.Series(dtype=object)
if universe_mode.startswith("S&P 500"):
    with st.spinner("Loading S&P 500 constituents..."):
        sector_map = _sp500_constituents()
    tickers_input = sector_map.index.tolist()
elif universe_mode.startswith("S&P 1500"):
    with st.spinner("Loading S&P 1500 constituents..."):
        sector_map = _sp1500_constituents()
    tickers_input = sector_map.index.tolist()
else:
    # Custom watchlist: best-effort sector lookup from the cached S&P 1500 map
    try:
        sector_map = _sp1500_constituents()
    except Exception:
        sector_map = pd.Series(dtype=object)

if not tickers_input:
    ui.banner("warn", "Specify at least one symbol.")
    st.stop()

tickers_key = tuple(sorted(tickers_input))
n_requested = len(tickers_key)

# ── Price retrieval (batched; cache-covered names skip the network) ──────────

prices_cache_key = (tickers_key, str(start_date), str(end_date))
needs_fetch = st.session_state.get("_prices_key") != prices_cache_key

if needs_fetch:
    n = n_requested
    prog = st.progress(0.0, text=f"Retrieving prices: 0 / {n}")

    frames = data.get_prices_batch(
        list(tickers_key), start_date, end_date,
        progress_cb=lambda done, total: prog.progress(
            min(done / total, 1.0), text=f"Retrieving prices: {done} / {total}"
        ),
    )
    prog.empty()

    price_dict = {
        t: df["adj_close"].rename(t)
        for t, df in frames.items()
        if not df.empty and "adj_close" in df.columns
    }
    dl_failed = sorted(set(tickers_key) - set(price_dict))

    price_df = pd.DataFrame(price_dict).sort_index() if price_dict else pd.DataFrame()

    st.session_state["_prices_key"] = prices_cache_key
    st.session_state["_price_df"]   = price_df
    st.session_state["_dl_failed"]  = dl_failed
else:
    price_df  = st.session_state["_price_df"]
    dl_failed = st.session_state["_dl_failed"]

if price_df.empty:
    ui.banner("error", "No price data was returned. Verify the symbols and history window.")
    st.stop()

if len(price_df) < 252:
    ui.banner(
        "warn",
        f"The selected window contains only {len(price_df)} sessions; 252 are "
        "required to compute all signals. Extend the history window start.",
    )
    st.stop()

active_tickers = tuple(sorted(price_df.columns.tolist()))

with st.spinner(f"Retrieving market capitalizations for {len(active_tickers)} symbols..."):
    mcaps = _market_caps(active_tickers)

with st.spinner("Retrieving revenue growth (best effort, up to 60 seconds)..."):
    rev_growth = _rev_growth(active_tickers)

with st.spinner("Computing signals and composite ranks..."):
    ranked, insufficient, suspect = _screen(price_df, mcaps, rev_growth, float(min_market_cap))

if max_extension < 5.0:
    mask = ranked["extension_z"].isna() | (ranked["extension_z"] <= max_extension)
    ranked = ranked[mask]

# Attach GICS sectors (free — same Wikipedia table as the constituents)
if not sector_map.empty:
    ranked = ranked.copy()
    ranked["sector"] = sector_map.reindex(ranked.index).fillna("—")

# Sector filter: a post-ranking slice, so changing it never re-runs the scan
sectors_present = (
    sorted(s for s in ranked.get("sector", pd.Series(dtype=object)).unique() if s and s != "—")
    if "sector" in ranked.columns else []
)
if sectors_present:
    sel_sectors = st.multiselect(
        "Sector Filter (GICS)", options=sectors_present, default=[],
        help="Restrict the ranking to selected sectors. Leave empty for all sectors.",
    )
    if sel_sectors:
        ranked = ranked[ranked["sector"].isin(sel_sectors)]

# ── Coverage summary ──────────────────────────────────────────────────────────

ui.kpi_row([
    {"label": "Universe Requested", "value": f"{n_requested}"},
    {"label": "Ranked", "value": f"{len(ranked)}"},
    {"label": "Insufficient History", "value": f"{len(insufficient)}"},
    {"label": "Data Integrity Exclusions", "value": f"{len(suspect)}"},
    {"label": "Retrieval Failures", "value": f"{len(dl_failed)}"},
])

# ── Ranked table ──────────────────────────────────────────────────────────────

with ui.panel("Composite Ranking"):
    if ranked.empty:
        ui.banner("info", "No symbols passed all constraints with sufficient history.")
    else:
        display_cols = [
            "composite", "sector",
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
            "composite":        st.column_config.NumberColumn("Composite Score", format="%.2f"),
            "sector":           st.column_config.TextColumn("GICS Sector"),
            "extension_z":      st.column_config.NumberColumn("Extension (sd)", format="%.2f",
                                    help="Standard deviations above the symbol's own "
                                         "252-session regression trendline"),
            "extension_flag":   st.column_config.TextColumn("Extension Status",
                                    help="On trend / Extended / Stretched"),
            "mom_12_1":         st.column_config.NumberColumn("12-1 Momentum", format="%.1f%%"),
            "mom_6m":           st.column_config.NumberColumn("6-Month Momentum", format="%.1f%%"),
            "pct_above_200sma": st.column_config.NumberColumn("vs 200-Day Average", format="%.1f%%"),
            "golden_cross":     st.column_config.CheckboxColumn("50 > 200",
                                    help="50-session average above 200-session average"),
            "dist_52w_high":    st.column_config.NumberColumn("vs 52-Week High", format="%.1f%%"),
            "trend_slope":      st.column_config.NumberColumn("Trend Slope (annualized)", format="%.1f%%"),
            "trend_r2":         st.column_config.NumberColumn("Trend R²", format="%.2f"),
            "trailing_vol":     st.column_config.NumberColumn("Realized Volatility", format="%.1f%%"),
            "market_cap_b":     st.column_config.NumberColumn("Market Cap ($B)", format="$%.1f"),
            "rev_growth_yoy":   st.column_config.NumberColumn("Revenue Growth YoY", format="%.1f%%"),
        }

        st.dataframe(
            disp.drop(columns=["market_cap"], errors="ignore"),
            column_config=col_cfg,
            width="stretch",
            height=520,
        )

        csv = ranked[[c for c in display_cols if c in ranked.columns]].reset_index().to_csv(index=False)
        st.download_button("Export Full Results (CSV)", csv,
                           file_name="screen_results.csv", mime="text/csv")

# ── Scatter ───────────────────────────────────────────────────────────────────

if not ranked.empty:
    with ui.panel("Momentum vs Trend Quality"):
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
                    "12-1 momentum: %{x:.1f}%<br>"
                    "Trend R²: %{y:.2f}<br>"
                    "Extension: %{customdata[2]} sd (%{customdata[3]})"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=sizes,
                    color=plot_df["composite"].tolist(),
                    colorscale="Plasma",
                    cmin=c_min, cmax=c_max,
                    colorbar=dict(title="Composite", thickness=14),
                    line=dict(width=0.5, color="rgba(237,234,227,0.5)"),
                ),
            ))

            top20 = set(plot_df.nlargest(20, "composite").index)
            scatter_fig.data[0].text = [t if t in top20 else "" for t in plot_df.index]

            scatter_fig.add_annotation(
                x=0.97, y=0.97, xref="paper", yref="paper",
                text="Strong, persistent uptrend", showarrow=False,
                font=dict(size=11, color="rgba(237,234,227,0.8)"),
                bgcolor="rgba(27,26,24,0.75)", borderpad=4,
            )
            scatter_fig.update_layout(
                xaxis_title="12-1 momentum (%)", yaxis_title="Trend R²",
                yaxis=dict(range=[-0.05, 1.05]),
                height=420, margin=dict(l=10, r=10, t=10, b=10),
                hovermode="closest",
            )
            apply_chart_theme(scatter_fig)
            st.plotly_chart(scatter_fig, width="stretch", config=CHART_CONFIG)
            st.caption(
                "Marker size and color both encode the composite rank score. The "
                "upper-right region combines strong trailing momentum with a smooth, "
                "persistent trend. The extension statistic shown on hover measures "
                "how far the latest price sits above the symbol's own trendline; "
                "highly extended names may represent stretched entry points."
            )

# ── Exclusions ────────────────────────────────────────────────────────────────

if not insufficient.empty:
    with st.expander(f"Insufficient Price History — {len(insufficient)} symbol(s)"):
        st.caption(
            "Fewer than 80% of the required 252 sessions in the trailing window, "
            "or insufficient pre-window history. Common causes: recent listings, "
            "spinoffs, exchange migrations."
        )
        st.dataframe(pd.DataFrame({"ticker": insufficient.index}).reset_index(drop=True),
                     width="stretch")

if not suspect.empty:
    with st.expander(f"Data Integrity Exclusions — {len(suspect)} symbol(s)"):
        st.caption(
            "Non-physical signal values (12-1 momentum above 500% or 6-month "
            "momentum above 300%) almost always reflect spinoff stub prices, "
            "unadjusted corporate actions, or reused symbols. These names were "
            "excluded before cross-sectional scoring to avoid distorting the "
            "remainder of the universe."
        )
        disp_suspect = suspect[["mom_12_1", "mom_6m"]].copy()
        disp_suspect["mom_12_1"] = disp_suspect["mom_12_1"] * 100
        disp_suspect["mom_6m"]   = disp_suspect["mom_6m"] * 100
        disp_suspect = disp_suspect.sort_values("mom_12_1", key=abs, ascending=False)
        disp_suspect.index.name = "ticker"
        st.dataframe(
            disp_suspect,
            column_config={
                "mom_12_1": st.column_config.NumberColumn("12-1 Momentum", format="%.1f%%"),
                "mom_6m":   st.column_config.NumberColumn("6-Month Momentum", format="%.1f%%"),
            },
            width="stretch",
        )

if dl_failed:
    with st.expander(f"Retrieval Failures — {len(dl_failed)} symbol(s)"):
        st.caption(
            "No price data after three attempts. Possible causes: delisting, "
            "symbol changes, or source unavailability."
        )
        st.dataframe(pd.DataFrame({"ticker": sorted(dl_failed)}).reset_index(drop=True),
                     width="stretch")

ui.footer_disclaimer()
