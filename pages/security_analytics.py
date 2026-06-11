"""Security Analytics — single-instrument performance, risk, and distribution diagnostics."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as scstats

import ui
from src import analysis, data
from src.theme import PRIMARY, BENCHMARK, PRIMARY_10, CHART_CONFIG, apply_chart_theme

ui.page_header(
    "Equity Research", "Security Analytics",
    "Total-return profile, drawdown history, return distribution, and rolling "
    "risk statistics for a single instrument, benchmarked against a reference index.",
)

# ── Parameters ────────────────────────────────────────────────────────────────

today = date.today()

with ui.panel("Parameters"):
    c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1.2])
    with c1:
        ticker = st.text_input("Instrument", value="AAPL").strip().upper()
    with c2:
        benchmark = st.text_input("Benchmark", value="SPY",
                                  help="Reference index for relative performance. "
                                       "Leave blank to omit.").strip().upper()
    with c3:
        start_date, end_date = ui.date_range_input(
            "Observation Window", today - timedelta(days=365 * 5), today,
        )
    with c4:
        rf = ui.rf_rate_input()

    c5, c6, c7 = st.columns([1.4, 1, 1.4])
    with c5:
        price_field = st.selectbox(
            "Price Basis", options=["adj_close", "close"], index=0,
            format_func={"adj_close": "Total return (adjusted close)",
                         "close": "Price return (unadjusted close)"}.get,
            help="Adjusted close incorporates splits and dividends and is the "
                 "standard basis for return computations.",
        )
    with c6:
        log_scale = st.checkbox("Logarithmic price axis", value=False)
    with c7:
        if st.button("Invalidate Local Cache"):
            n = data.clear_cache(ticker)
            ui.banner("success", f"Removed <span class='mono'>{n}</span> cached file(s) for {ticker}.")

if not ticker:
    ui.banner("info", "Enter an instrument symbol to begin.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return data.get_prices(ticker, start, end)


with st.spinner(f"Retrieving {ticker}..."):
    df = fetch(ticker, start_date, end_date)

if df.empty:
    ui.banner("error", f"No price data for <b>{ticker}</b>. Verify the symbol or adjust the observation window.")
    st.stop()

bench_df = pd.DataFrame()
if benchmark and benchmark != ticker:
    with st.spinner(f"Retrieving benchmark {benchmark}..."):
        bench_df = fetch(benchmark, start_date, end_date)

prices = df[price_field].dropna()
returns = analysis.simple_returns(prices)

bench_returns = pd.Series(dtype=float)
if not bench_df.empty:
    bench_returns = analysis.simple_returns(bench_df[price_field].dropna())

# ── Headline ──────────────────────────────────────────────────────────────────

total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
ui.kpi_row([
    {"label": "Last Price", "value": f"${prices.iloc[-1]:,.2f}"},
    {"label": "Period Total Return", "value": f"{total_ret:+,.1f}%",
     "delta_kind": "pos" if total_ret >= 0 else "neg"},
    {"label": "Annualized Return", "value": f"{analysis.annualized_return(returns) * 100:,.2f}%"},
    {"label": "Annualized Volatility", "value": f"{analysis.annualized_volatility(returns) * 100:,.2f}%"},
    {"label": "Sharpe Ratio", "value": f"{analysis.sharpe_ratio(returns, rf=rf):.2f}"},
])

# ── Price history ─────────────────────────────────────────────────────────────

with ui.panel("Price History"):
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=prices.index, y=prices.values, name=ticker,
        line=dict(color=PRIMARY, width=1.5),
    ))
    if not bench_df.empty:
        bench_prices = bench_df[price_field].dropna()
        rescale = prices.iloc[0] / bench_prices.iloc[0]
        fig_price.add_trace(go.Scatter(
            x=bench_prices.index, y=bench_prices * rescale,
            name=f"{benchmark} (rebased)",
            line=dict(color=BENCHMARK, width=1, dash="dot"), opacity=0.7,
        ))
    fig_price.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
    )
    apply_chart_theme(fig_price)
    st.plotly_chart(fig_price, width="stretch", config=CHART_CONFIG)

# ── Cumulative return and drawdown ────────────────────────────────────────────

dd = analysis.drawdown(returns)
wealth = analysis.cumulative_returns(returns)

with ui.panel("Cumulative Return and Drawdown"):
    fig_cum = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.05,
        subplot_titles=("Growth of $1", "Drawdown"),
    )
    fig_cum.add_trace(
        go.Scatter(x=wealth.index, y=wealth.values, name=ticker,
                   line=dict(color=PRIMARY, width=1.5)),
        row=1, col=1,
    )
    if len(bench_returns):
        bench_wealth = analysis.cumulative_returns(bench_returns)
        fig_cum.add_trace(
            go.Scatter(x=bench_wealth.index, y=bench_wealth.values, name=benchmark,
                       line=dict(color=BENCHMARK, width=1, dash="dot"), opacity=0.7),
            row=1, col=1,
        )
    fig_cum.add_trace(
        go.Scatter(x=dd.series.index, y=dd.series.values * 100,
                   name="Drawdown %", fill="tozeroy",
                   fillcolor=PRIMARY_10,
                   line=dict(color=PRIMARY, width=0.5)),
        row=2, col=1,
    )
    fig_cum.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                          hovermode="x unified", showlegend=True)
    fig_cum.update_yaxes(title_text="$ (log)", type="log", row=1, col=1)
    fig_cum.update_yaxes(title_text="%", row=2, col=1)
    apply_chart_theme(fig_cum)
    st.plotly_chart(fig_cum, width="stretch", config=CHART_CONFIG)
    st.caption(
        f"Maximum drawdown {dd.max_drawdown * 100:,.1f}%: "
        f"peak {dd.peak_date.date()}, trough {dd.trough_date.date()}."
    )

# ── Risk and performance summary ──────────────────────────────────────────────

col_l, col_r = st.columns([1, 1])

with col_l:
    with ui.panel("Performance and Risk Summary"):
        summary = analysis.summary_table(returns, rf=rf)

        def fmt(v, name):
            if pd.isna(v):
                return "—"
            if "ratio" in name.lower() or "kurtosis" in name.lower() or "skewness" in name.lower():
                return f"{v:.3f}"
            return f"{v * 100:.2f}%"

        summary["Value"] = [fmt(v, idx) for idx, v in zip(summary.index, summary["Value"])]
        st.dataframe(summary, width="stretch")

with col_r:
    with ui.panel("Distribution Diagnostics"):
        dist = analysis.distribution_stats(returns)
        rows = {
            "Observations": f"{dist['n_obs']:,}",
            "Mean (daily)": f"{dist['mean_daily'] * 100:.3f}%",
            "Std deviation (daily)": f"{dist['stdev_daily'] * 100:.3f}%",
            "Skewness": f"{dist['skewness']:.3f}",
            "Excess kurtosis": f"{dist['kurtosis_excess']:.3f}",
            "Jarque-Bera p-value": f"{dist['jarque_bera_p']:.4f}",
            "Min / Max (daily)": f"{dist['min'] * 100:.2f}% / {dist['max'] * 100:.2f}%",
        }
        st.dataframe(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]),
                     width="stretch")
        if dist["jarque_bera_p"] < 0.05:
            st.caption(
                "Jarque-Bera rejects normality at the 5% level. Equity returns "
                "routinely fail normality tests; models assuming normal returns — "
                "including parametric VaR — tend to understate tail risk."
            )

# ── Return distribution ───────────────────────────────────────────────────────

with ui.panel("Return Distribution vs Fitted Normal"):
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns.values * 100, nbinsx=80,
        name="Observed daily returns",
        histnorm="probability density", opacity=0.75, marker_color=PRIMARY,
    ))
    x_grid = np.linspace(returns.min(), returns.max(), 400)
    normal_pdf = scstats.norm.pdf(x_grid, loc=returns.mean(), scale=returns.std(ddof=1))
    fig_hist.add_trace(go.Scatter(
        x=x_grid * 100, y=normal_pdf / 100,
        name="Fitted normal", line=dict(color=BENCHMARK, width=2),
    ))
    fig_hist.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Daily return (%)", yaxis_title="Density",
    )
    apply_chart_theme(fig_hist)
    st.plotly_chart(fig_hist, width="stretch", config=CHART_CONFIG)

# ── Rolling statistics ────────────────────────────────────────────────────────

with ui.panel("Rolling 60-Day Statistics"):
    window = 60
    rolling_vol = returns.rolling(window).std(ddof=1) * np.sqrt(analysis.TRADING_DAYS) * 100
    rolling_sharpe = (
        (returns.rolling(window).mean() - rf / analysis.TRADING_DAYS)
        / returns.rolling(window).std(ddof=1)
    ) * np.sqrt(analysis.TRADING_DAYS)

    fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=("Annualized volatility (%)", "Annualized Sharpe ratio"))
    fig_roll.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Volatility",
                                  line=dict(color=PRIMARY, width=1.2)), row=1, col=1)
    fig_roll.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Sharpe",
                                  line=dict(color=BENCHMARK, width=1.2)), row=2, col=1)
    fig_roll.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                           hovermode="x unified", showlegend=False)
    fig_roll.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    apply_chart_theme(fig_roll)
    st.plotly_chart(fig_roll, width="stretch", config=CHART_CONFIG)

with st.expander("Raw OHLCV Data (last 250 sessions)"):
    st.dataframe(df.tail(250), width="stretch")

ui.footer_disclaimer()
