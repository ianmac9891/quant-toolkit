"""Stock analysis: pull a ticker, show price, returns, risk metrics, distribution."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as scstats

from src import analysis, data
from src.theme import PRIMARY, BENCHMARK, apply_chart_theme


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
benchmark = st.sidebar.text_input("Benchmark (optional)", value="SPY").strip().upper()

today = date.today()
default_start = today - timedelta(days=365 * 5)

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(default_start, today),
    min_value=date(1990, 1, 1),
    max_value=today,
)

rf_pct = st.sidebar.number_input(
    "Risk-free rate (% annual)",
    min_value=0.0, max_value=20.0, value=4.5, step=0.25,
    help="Used to compute the Sharpe and Sortino ratios. A common proxy is the current 3-month T-bill yield.",
)
rf = rf_pct / 100

price_field = st.sidebar.selectbox(
    "Price series",
    options=["adj_close", "close"],
    index=0,
    help="Adjusted close accounts for splits and dividends and is the standard input for return calculations. Raw close is the unadjusted price.",
)

log_scale = st.sidebar.checkbox("Log scale on price chart", value=False)

if st.sidebar.button("Clear cache for this ticker"):
    n = data.clear_cache(ticker)
    st.sidebar.success(f"Cleared {n} cached file(s) for {ticker}")


# ---------------------------------------------------------------------------
# Data fetch (cached at the Streamlit layer too)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return data.get_prices(ticker, start, end)


with st.spinner(f"Fetching {ticker}..."):
    df = fetch(ticker, start_date, end_date)

if df.empty:
    st.error(f"No data for {ticker}. Check the ticker symbol or try a different date range.")
    st.stop()

bench_df = pd.DataFrame()
if benchmark and benchmark != ticker:
    with st.spinner(f"Fetching benchmark {benchmark}..."):
        bench_df = fetch(benchmark, start_date, end_date)

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

prices = df[price_field]
returns = analysis.simple_returns(prices)
log_rets = analysis.log_returns(prices)

bench_returns = pd.Series(dtype=float)
if not bench_df.empty:
    bench_returns = analysis.simple_returns(bench_df[price_field])

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(f"{ticker}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest price", f"${prices.iloc[-1]:,.2f}")
col2.metric(
    "Total return",
    f"{(prices.iloc[-1] / prices.iloc[0] - 1) * 100:,.1f}%",
)
col3.metric("Annualized return", f"{analysis.annualized_return(returns) * 100:,.2f}%")
col4.metric("Annualized volatility", f"{analysis.annualized_volatility(returns) * 100:,.2f}%")

# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------

st.subheader("Price")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=prices.index, y=prices.values, name=ticker,
    line=dict(color=PRIMARY, width=1.5),
))
if not bench_df.empty:
    rescale = prices.iloc[0] / bench_df[price_field].iloc[0]
    fig_price.add_trace(
        go.Scatter(
            x=bench_df.index,
            y=bench_df[price_field] * rescale,
            name=f"{benchmark} (rescaled)",
            line=dict(color=BENCHMARK, width=1, dash="dot"),
            opacity=0.7,
        )
    )
fig_price.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_type="log" if log_scale else "linear",
    hovermode="x unified",
)
apply_chart_theme(fig_price)
st.plotly_chart(fig_price, use_container_width=True, config={"responsive": True, "displayModeBar": False})

# ---------------------------------------------------------------------------
# Cumulative return + drawdown
# ---------------------------------------------------------------------------

st.subheader("Cumulative return and drawdown")

dd = analysis.drawdown(returns)
wealth = analysis.cumulative_returns(returns)

fig_cum = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.05,
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
    go.Scatter(
        x=dd.series.index, y=dd.series.values * 100,
        name="Drawdown %", fill="tozeroy",
        line=dict(color=PRIMARY, width=0.5),
    ),
    row=2, col=1,
)
fig_cum.update_layout(
    height=400, margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified", showlegend=True,
)
fig_cum.update_yaxes(title_text="$ (log)", type="log", row=1, col=1)
fig_cum.update_yaxes(title_text="%", row=2, col=1)
apply_chart_theme(fig_cum)
st.plotly_chart(fig_cum, use_container_width=True, config={"responsive": True, "displayModeBar": False})

st.caption(
    f"Max drawdown: **{dd.max_drawdown * 100:,.1f}%**, "
    f"peak {dd.peak_date.date()}, trough {dd.trough_date.date()}."
)

# ---------------------------------------------------------------------------
# Risk metrics table
# ---------------------------------------------------------------------------

st.subheader("Risk and performance metrics")

summary = analysis.summary_table(returns, rf=rf)
def fmt(v, name):
    if pd.isna(v):
        return "—"
    if "ratio" in name.lower() or "kurtosis" in name.lower() or "skewness" in name.lower():
        return f"{v:.3f}"
    return f"{v * 100:.2f}%"

summary["Value"] = [fmt(v, idx) for idx, v in zip(summary.index, summary["Value"])]
st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

st.subheader("Return distribution")

dist = analysis.distribution_stats(returns)

col_a, col_b = st.columns([2, 1])

with col_a:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=80,
        name="Observed daily returns (%)",
        histnorm="probability density",
        opacity=0.75,
        marker_color=PRIMARY,
    ))
    x_grid = np.linspace(returns.min(), returns.max(), 400)
    normal_pdf = scstats.norm.pdf(x_grid, loc=returns.mean(), scale=returns.std(ddof=1))
    fig_hist.add_trace(go.Scatter(
        x=x_grid * 100, y=normal_pdf / 100,
        name="Fitted normal",
        line=dict(color=BENCHMARK, width=2),
    ))
    fig_hist.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Daily return (%)",
        yaxis_title="Density",
    )
    apply_chart_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True, config={"responsive": True, "displayModeBar": False})

with col_b:
    st.markdown("**Distribution stats**")
    rows = {
        "Observations": f"{dist['n_obs']:,}",
        "Mean (daily)": f"{dist['mean_daily'] * 100:.3f}%",
        "Stdev (daily)": f"{dist['stdev_daily'] * 100:.3f}%",
        "Skewness": f"{dist['skewness']:.3f}",
        "Excess kurtosis": f"{dist['kurtosis_excess']:.3f}",
        "Jarque-Bera p-value": f"{dist['jarque_bera_p']:.4f}",
        "Min / Max": f"{dist['min'] * 100:.2f}% / {dist['max'] * 100:.2f}%",
    }
    st.dataframe(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]),
                 use_container_width=True)
    if dist["jarque_bera_p"] < 0.05:
        st.caption(
            "Jarque-Bera rejects normality (p < 0.05). Equity returns routinely fail "
            "normality tests; models that assume a normal distribution — including "
            "parametric VaR — will tend to understate tail risk."
        )

# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------

st.subheader("Rolling 60-day stats")

window = 60
rolling_vol = returns.rolling(window).std(ddof=1) * np.sqrt(analysis.TRADING_DAYS) * 100
rolling_sharpe = (
    (returns.rolling(window).mean() - rf / analysis.TRADING_DAYS)
    / returns.rolling(window).std(ddof=1)
) * np.sqrt(analysis.TRADING_DAYS)

fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("Annualized volatility (%)", "Annualized Sharpe"))
fig_roll.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Vol",
                              line=dict(color=PRIMARY, width=1.2)), row=1, col=1)
fig_roll.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Sharpe",
                              line=dict(color=BENCHMARK, width=1.2)), row=2, col=1)
fig_roll.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                       hovermode="x unified", showlegend=False)
fig_roll.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
apply_chart_theme(fig_roll)
st.plotly_chart(fig_roll, use_container_width=True, config={"responsive": True, "displayModeBar": False})

# ---------------------------------------------------------------------------
# Raw data expander
# ---------------------------------------------------------------------------

with st.expander("Raw OHLCV data"):
    st.dataframe(df.tail(250), use_container_width=True)
