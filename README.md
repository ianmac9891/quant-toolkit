# Quant Research Terminal

A sidebar-free Streamlit research terminal for equity, derivatives, portfolio,
and macro analysis. Pure-Python analytics live in `src/` and the UI is a thin
layer on top, so every model is importable from a notebook or script without
touching Streamlit.

<!-- TODO: add a home-page screenshot here (no screenshots/ dir exists yet;
     capture the landing page at ~1280px wide and commit it under docs/). -->

## Tools

**Equity & Derivatives Research**
- **Security Analytics** - total-return profile, drawdowns, return distribution, benchmark regression, monthly return heatmap, rolling risk statistics.
- **Volatility Analytics** - GARCH and GJR-GARCH conditional volatility, bootstrap-simulated price distributions, regime classification.
- **Event Study** - market-model abnormal returns with Brown-Warner, Patell, and BMP test statistics; multi-event aggregation.
- **Earnings Move Analysis** - historical earnings-day reactions with the current option-implied move shown against the realized distribution.
- **Derivatives Workbench** - multi-leg Black-Scholes-Merton position modeling: payoff profiles, Greeks, probability of profit, IV solver.
- **Options Chain Explorer** - listed chains from delayed Yahoo data: IV smile and term structure, straddle-implied move, open interest and volume positioning, quote table with model deltas.

**Portfolio & Risk**
- **Portfolio Construction** - maximum Sharpe, minimum variance, and risk parity with Ledoit-Wolf and OAS shrinkage, James-Stein means, Michaud resampling, efficient frontier.
- **Risk Analytics** - historical and parametric VaR and CVaR in dollars at selectable horizons, Monte Carlo wealth simulation, Fama-French three-factor exposure, stress-window replay.
- **Strategy Simulation** - no-lookahead backtesting with transaction costs, walk-forward optimization, estimation versus validation segmentation.
- **Correlation Analytics** - correlation matrix, rolling diversification pulse, PC1 variance share, diversification ratio.

**Systematic Research**
- **Equity Screening** - cross-sectional momentum and trend-quality ranking across the S&P 500 or S&P 1500 with GICS sector filters and data-integrity exclusions.
- **Seasonality Research** - calendar-effect hypothesis testing with Benjamini-Hochberg FDR control and out-of-sample replication requirements.
- **Relative Value Analysis** - pair cointegration: hedge ratio, Engle-Granger test, spread half-life, z-score bands.

**Macro & Rates**
- **Yield Curve Monitor** - Treasury curve snapshot and history, inversion-tracking spreads; exports the current bill rate as the app-wide risk-free default.

## Data sources

- **Yahoo Finance** via yfinance (daily OHLCV, option chains, earnings dates, Treasury yield indices). No key required; quotes are delayed and indicative.
- **Alpha Vantage** as a single-ticker price fallback (free tier, 25 requests/day; key in `.env`).
- **Ken French Data Library** for daily factor returns.

Prices are cached to Parquet in `cache/` (gitignored) with an in-memory layer
on top, so reruns and large screens stay off the network.

---

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Alpha Vantage key (optional but recommended)
```

## Running

```bash
source .venv/bin/activate
streamlit run app.py
```

A browser tab opens at http://localhost:8501. There is no sidebar: tools are
selected from the landing page, and each page's top bar routes back to it.

## Tests

```bash
pytest tests/ -v
```

Reference-value tests, all offline: Black-Scholes-Merton against Hull's
worked examples, put-call parity as an identity, implied-vol round-trips,
event-study statistics on a hand-derivable synthetic fixture, pair half-life
on a simulated OU process, and cache-staleness logic with frozen dates.

## Project layout

```
quant/
├── app.py            # st.navigation registry (hidden nav), design system injection
├── home.py           # landing page with tool cards
├── ui.py             # Streamlit-side design system, cached data access, session context
├── pages/            # one thin script per tool
├── src/              # pure-Python library: no Streamlit imports anywhere
├── tests/            # offline reference-value tests (pytest)
├── cache/            # Parquet price cache (gitignored)
└── .streamlit/       # theme config
```

`src/` is the library; `pages/` is the UI. Adding a tool means a new
`pages/<name>.py` registered in `app.py` with a `url_path` plus a nav card on
`home.py` (see CLAUDE.md for the full conventions).

## When yfinance breaks

It will; Yahoo changes endpoints periodically. `pip install --upgrade
yfinance` usually fixes it within days. The app requires `curl_cffi` so
yfinance can impersonate a browser TLS fingerprint, which is what keeps the
shared Streamlit Cloud egress IP from being rate-limited.
