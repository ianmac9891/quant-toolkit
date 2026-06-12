# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the virtualenv first (required every new terminal session)
source .venv/bin/activate

# Run the app
streamlit run app.py

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_analysis.py::test_sharpe_nonzero -v

# Force-refresh cached data for a ticker (delete parquet files)
python -c "from src.data import clear_cache; clear_cache('AAPL')"
```

## Architecture

The app is the **Quant Research Terminal (QRT)** — a sidebar-free Streamlit app. `app.py` registers all pages with `st.navigation(position="hidden")`; `home.py` is the landing page with HTML navigation cards (rendered via `st.html` — block-level HTML inside an `<a>` breaks `st.markdown`'s parser); each tool page carries a top bar that routes back to `/`.

**Layer separation is the core design principle.** `src/` is a pure Python library — no Streamlit imports. `pages/` is thin UI that imports from `src/`. This means analysis functions can be used from notebooks or scripts without touching Streamlit.

**`ui.py` (root level)** is the Streamlit-side design system, kept out of `src/` to preserve that rule:
- `inject_design_system()` — the full CSS (fonts, panels, inputs, custom `qrt-*` classes); injected once in `app.py`
- `page_header(section, title, description)` — top bar + page title; required on every page
- `panel(title)` — context manager wrapping `st.container(border=True)` with a kicker label
- `kpi_row(items)`, `banner(kind, body)`, `tag(text, kind)`, `nav_card(...)`, `footer_disclaimer()`
- `date_range_input(...)` — guards the mid-selection state of range `st.date_input` (one date picked → would crash on tuple unpack); **always use this for date ranges**
- `rf_rate_input()` — the single risk-free convention: entered in percent, returned as decimal; `get_default_rf_pct()`/`set_default_rf_pct()` carry the app-wide rate default written by the Yield Curve Monitor
- **Data access lives here, not in pages**: `ui.fetch_prices(ticker, start, end)` (single name, returns `data.PriceResult`, never raises) and `ui.fetch_universe(tuple, start, end)` (batch) are the `@st.cache_data(ttl=3600)` wrappers over `src/data.py`. Pages must not define their own raw-price cache wrappers; derived-computation caches stay page-local. On failure render `ui.data_unavailable(detail)`; every page showing price-derived output ends its fetch with `ui.data_asof_caption(asof, source)`
- Session context: `get_default_ticker`/`remember_ticker` and `get_default_universe`/`remember_universe` make the last-used instrument and universe follow the analyst across tools
- `download_row(df, filename_stem)` — right-aligned tertiary CSV export; required under every results table
- Pages never call `st.metric` / `st.info` / `st.warning` / `st.error` directly — use `kpi_row` and `banner`. No emojis or Material icons anywhere.

**Pages** (registered in `app.py` with stable `url_path`s):
- Equity & Derivatives Research: `security_analytics.py`, `volatility_analytics.py`, `event_study.py`, `earnings_analysis.py`, `derivatives_workbench.py`, `options_chain.py`
- Portfolio & Risk: `portfolio_construction.py`, `risk_analytics.py`, `strategy_simulation.py`, `correlation_analytics.py`
- Systematic Research: `equity_screening.py`, `seasonality_research.py`, `relative_value.py`
- Macro & Rates: `yield_curve.py` (also publishes the 13-week bill rate as the app-wide risk-free default via `ui.set_default_rf_pct`)
- Expensive pages (construction, simulation, screening) wrap parameters in `st.form` to batch reruns. Portfolio Construction stages its result in `st.session_state["portfolio_weights"/"portfolio_prices"/"portfolio_returns"/"portfolio_method"/"portfolio_cov"]` for Risk Analytics.

**Data flow:**
1. `src/data.py` defines a `Provider` ABC with `get_prices(ticker, start, end) -> DataFrame`.
2. `YFinanceProvider` is the primary implementation; `AlphaVantageProvider` is a fallback for fundamentals (25 req/day free tier — use sparingly).
3. `fetch_prices(ticker, start, end) -> PriceResult` is the resilient single-name entry point: yfinance, one retry with backoff, Alpha Vantage fallback (never used for universes), typed result with `ok`/`error`/`asof`/`source`. yfinance ≥ 1.4 with curl_cffi manages its own browser-impersonating session — never pass a custom `requests.Session`.
4. `get_prices()` (the module-level function) wraps providers with a parquet cache in `cache/`. It loads from cache, checks if the requested date range is covered, and only hits the network for missing data. Delete files in `cache/` to force a refresh.
5. Cache-coverage checks are **business-day clamped** (`_cache_is_stale`): a requested end of today/weekend is satisfied by a cache ending on the prior completed business day, and a weekend start by data beginning the following Monday. Without this every ticker refetches on every run, which is what rate-limits large scans.
6. `get_prices_batch(tickers, start, end, progress_cb)` is the **universe-scale path** (used by Equity Screening and Correlation Analytics): cache-covered names skip the network entirely; stale names are batch-downloaded via chunked `yf.download(group_by="ticker")` (~100 per request) and merged back into the per-ticker parquet cache. Never loop single-name `get_prices` over hundreds of tickers — that is the rate-limited pattern that drops names.
7. Pages access prices through `ui.fetch_prices` / `ui.fetch_universe` (the cached wrappers) and pass the result to `src/` analytics.

**`src/portfolio.py`** provides portfolio optimization:
- `expected_returns(returns_df)`, `covariance_matrix(returns_df)` — build inputs from a returns DataFrame
- `portfolio_stats(weights, mu, cov, rf)` — compute (return, vol, Sharpe) for any weight vector
- `max_sharpe(mu, cov, rf, weight_cap)` — Markowitz/Lintner transform → convex QP (cvxpy/CLARABEL)
- `min_variance(mu, cov, rf, weight_cap)` — standard QP; unconstrained when weight_cap=1.0
- `risk_parity(mu, cov, rf)` — log-barrier minimized via scipy L-BFGS-B; unconstrained by design
- `efficient_frontier(mu, cov, n_points, weight_cap)` — parametric QP sweep returning `DataFrame[volatility, expected_return]`

**`src/estimators.py`** provides robust parameter estimators for use with `portfolio.py`:
- Covariance: `ledoit_wolf_covariance`, `oas_covariance`, `sample_covariance` (all annualized; keyed in `COV_ESTIMATORS`)
- Means: `james_stein_mean` (shrinks toward grand mean), `sample_mean` (keyed in `MEAN_ESTIMATORS`)
- `resampled_weights(returns, method, rf, weight_cap, cov_estimator, mean_estimator)` — Michaud-style bootstrap × 200, average weight vectors

**`src/risk.py`** provides portfolio risk analysis:
- `portfolio_var(weights, returns_df)` → `VaRResult` with historical/parametric VaR and CVaR at 95%/99%
- `monte_carlo_paths(weights, returns_df, n_paths, horizon_days)` → wealth path array `(n_paths, horizon+1)` via iid bootstrap
- `load_ff3_factors(start, end)` — direct HTTP + zipfile download from Ken French's library; returns decimal-unit `DataFrame[Mkt-RF, SMB, HML, RF]`
- `factor_exposure(weights, prices_df, ff3)` → `FactorResult` (FF3 OLS regression via statsmodels)
- `stress_test(weights, prices_df)` → `list[StressResult]` over three predefined windows (2008 GFC, 2020 COVID, 2022 rate hikes)

**`src/backtest.py`** provides the backtesting engine:
- `StrategyFn = Callable[[pd.DataFrame], pd.Series]` — receives `prices.iloc[:i+1]`, returns target weights or empty Series for cash
- `run_backtest(prices, strategy_fn, initial_capital, rebalance_freq, cost_bps)` → `BacktestResult`
  - **No-lookahead**: signal at close of day t, weights apply from day t+1
  - **Weight drift**: after each day's return the engine drifts `current_weights` via `w * (1+r) / sum(w*(1+r))`; turnover at rebalance is measured from drifted weights, not prior targets
  - `rebalance_freq`: "D" / "W" / "M" / "Q" — uses last actual trading day of each period
- `perf_stats(equity, trade_log, rf)` → dict with Ann. return, vol, Sharpe, Sortino, Max drawdown, Calmar, Avg daily turnover

**`src/strategies.py`** provides strategy factories (all return a `StrategyFn`):
- `buy_and_hold(weights)` — fixed or equal weights; rebalances back to target at each period
- `ma_crossover(fast, slow)` — equal-weight assets whose fast MA > slow MA; cash when none qualify
- `cross_sectional_momentum(lookback_months, skip_months, top_k)` — Jegadeesh-Titman; uses ~21 days/month
- `walk_forward_optimizer(lookback_months, method, rf, weight_cap, cov_estimator, mean_estimator, min_obs)` — rolling MVO/risk-parity re-fit at each rebalance; falls back to equal weight when history < `min_obs`

**`src/analysis.py`** provides stateless functions operating on `pd.Series`:
- Returns: `simple_returns`, `log_returns`, `cumulative_returns`
- Performance: `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`
- Tail risk: `historical_var`, `historical_cvar`, `parametric_var`
- Diagnostics: `drawdown` (returns a `DrawdownResult` dataclass), `distribution_stats`, `summary_table`

**Return type convention:** Use simple returns when combining assets (portfolio math); use log returns for single-series statistics. `TRADING_DAYS = 252` is the annualization constant throughout.

**`src/screener.py`** provides the bullish stock screener:
- `fetch_sp500_constituents()` / `fetch_sp400_…` / `fetch_sp600_…` / `fetch_sp1500_…` — scrape Wikipedia constituents tables, returning `Series(GICS sector, index=ticker)`; normalizes BRK.B → BRK-B. `fetch_*_tickers()` wrappers return just the list. The screening page attaches the sector column and offers a post-ranking sector filter (a slice — never re-runs the scan)
- `fetch_ticker_prices(ticker, start, end, max_retries)` — per-ticker fetch via `data.get_prices` (parquet-cached) with exponential-backoff retry
- `compute_signals(prices)` → DataFrame of 7 signals per ticker: `mom_12_1`, `mom_6m`, `pct_above_200sma`, `golden_cross`, `dist_52w_high`, `trend_slope`, `trend_r2`; plus informational `extension_z`/`extension_flag`. Two exclusion rules: < 80% coverage in the trailing 252-day window (`_HISTORY_MIN=202`), OR fewer than `TRADING_DAYS + _HISTORY_BUFFER` (342) total valid rows in the full fetch window — the second rule catches recent spinoffs/IPOs whose stub prices would corrupt the momentum reference at t−252
- `score_and_rank(signals)` → cross-sectional z-scores (nanmean/nanstd) + equal-weight `composite`; NaN-aware so sparse tickers don't corrupt universe statistics
- `fetch_market_caps(tickers)` — `fast_info.market_cap` via ThreadPoolExecutor; fast, gates the market-cap filter
- `fetch_rev_growth(tickers, total_timeout=60)` — `yf.Ticker.info` best-effort, hard timeout; NaN on failure
- `trailing_volatility(prices)` — 252-day annualized vol from price data (no extra API calls)
- `run_screen(prices, market_caps, rev_growth, min_market_cap)` → `(ranked, insufficient_history, suspect_data)` — suspect_data contains tickers excluded for non-physical signals (mom_12_1 > 500% or mom_6m > 300%) before z-scoring; these are reported separately from insufficient_history on the page

**`src/anomalies.py`** provides the calendar anomaly lab:
- `run_anomaly_lab(prices, cost_bps)` → `list[AnomalyCategory]` — runs all 5 categories, applies BH-FDR correction across all ~20 hypotheses in a single pass, then sets `verdict` and `tradable` on each `HypothesisResult`
- Primary metric: **green-day rate** (proportion of days with return > 0); test: `proportions_ztest` from statsmodels
- Verdict: "Real pattern" iff `p_fdr < 0.05` AND sign(OOS gap) == sign(IS gap) AND |OOS gap| ≥ 0.5×|IS gap|; `tradable` is a separate secondary flag (`post_cost_mean_return_bps > 0`)
- 5 labeler functions: `label_day_of_week`, `label_month_of_year`, `label_turn_of_month`, `label_moon_phase`, `label_pre_holiday`
- Lunar phase uses synodic-month approximation from a known new-moon epoch (no `ephem` dependency); ±3-day window around new/full moon
- Pre-holiday uses `pandas_market_calendars` (NYSE calendar) with a rule-based fallback if not installed
- `BucketStats`: `green_rate`, `se_green`, `mean_bps`; `HypothesisResult`: all primary green-rate fields plus secondary `mean_return_bps`, `post_cost_mean_return_bps`

**`src/volforecast.py`** provides GARCH/GJR-GARCH volatility forecasting:
- `fit_garch(prices, model_type="garch"|"gjr")` → `GarchFit` — fits on daily **log returns × 100** (not simple returns; log space ensures zero-drift gives flat median); raises `ValueError` if persistence ≥ 1 (non-stationary). GJR adds the leverage term `γ·ε²·I(ε<0)`; its persistence is `α + β + γ/2`
- `simulate_paths(fit, current_price, horizon, drift_annual, n_sim, seed)` → `VolForecast` — bootstrap resamples empirical standardized residuals (preserves fat tails); initializes variance at `h_{t+1}` (one-step-ahead), not `h_t`; the recursion includes the GJR gamma term (zero for symmetric GARCH); returns 7 percentile paths (p2.5/10/25/50/75/90/97.5) plus `terminal_prices` array for O(1) live probability queries
- `analytic_vol_path(fit, horizon)` → `np.ndarray` — closed-form variance forecast starting from `h_{t+1}`: `h_{t+k} = h_lr + persistence^(k-1) × (h_{t+1} - h_lr)`; converges to long-run vol
- `p_above(forecast, target)` → `float` — `(terminal_prices > target).mean()`, called outside cache for live reference-level updates
- `GarchFit`: stores `omega`, `alpha`, `beta`, `gamma`, `model`, `persistence`, `h_current_pct2`, `h_next_pct2`, `h_lr_pct2`, annualized vols, `vol_regime` ("elevated"/"normal"/"compressed"), AIC, `std_resid`
- Requires `arch>=6.3.0` (in requirements.txt)

**`src/event_study.py`** test statistics: Brown-Warner naive t (constant variance), Patell Z (`_patell_stats` — prediction-error-corrected SARs; `MarketModelFit` carries `mkt_mean`/`mkt_ss` for the C_t correction), and for multi-event runs the BMP (1991) cross-sectional t on Patell-scaled SCARs (robust to event-induced variance).

**`src/options.py`** conventions: `prob_of_profit` simulates the underlying with a **single volatility** (`underlying_vol`, default `atm_leg_iv` — the leg nearest the money), never an average across legs. `payoff_bounds` determines unbounded profit/loss **analytically** from the net share exposure as S → ∞ (downside is always bounded at S=0) and reads bounded extremes off a dense grid. All terminal metrics (`breakevens`, `payoff_bounds`, `prob_of_profit`) evaluate at `eval_horizon_dte` — the **front** expiration — because a terminal-price model cannot settle expired legs path-dependently. `implied_vol` is Newton with a bisection fallback.

**`src/estimators.py`**: `james_stein_mean(returns, shrinkage=None)` uses the data-driven intensity `w = min(1, (k−2)·σ̄²/‖μ−μ̄1‖²)` when `shrinkage` is None; pass a float to override.

**`src/pairs.py`** (Relative Value Analysis): `analyze_pair(prices_a, prices_b)` → `PairResult` — log-price OLS hedge ratio, Engle-Granger cointegration (`coint`, the primary criterion because it accounts for the estimated hedge ratio), reference ADF on the spread, AR(1) half-life (`−ln2/ln(1+b)`, NaN unless −1 < b < 0), and a full-sample spread z-score. Raises `ValueError` below 120 overlapping sessions.

**`src/correlation.py`** (Correlation Analytics): `correlation_matrix`, `mean_offdiag_correlation`, `rolling_mean_correlation(returns, window)` (average pairwise correlation per date — the diversification pulse), `rolling_pair_correlation`, `pc1_variance_share` (top eigenvalue share of the correlation matrix), `diversification_ratio` (Choueifaty-Coignard), `extreme_pairs`.

**`src/theme.py`** is the single source of truth for design tokens and chart styling (pure Python, no Streamlit):
- The visual identity is **warm graphite + amber**: surfaces (`CANVAS`, `SURFACE`, `BORDER`, …) carry no blue cast, `PRIMARY` is amber (also the UI accent), `BENCHMARK` is muted steel blue, `POSITIVE`/`NEGATIVE` are muted green/red, `HEAT_NEG`/`HEAT_POS` are the diverging-heatmap endpoints. Opacity variants: `PRIMARY_10/18/28/80`
- Fonts: `FONT_DISPLAY` (Newsreader serif — page titles, masthead, card titles), `FONT_UI` (Inter — body, labels), `FONT_MONO` (IBM Plex Mono — numerals only). Never put body text or input copy in mono
- Never hardcode a hex color in a page — import the token. Banner semantics: info = steel, warn = amber, error = red, success = green
- `apply_chart_theme(fig)` — transparent surfaces, hairline grid, mono tick labels; uses `update_xaxes`/`update_yaxes` so it works on multi-subplot figures. It only styles `title_font` when a title exists — Plotly 6 renders the literal string "undefined" otherwise
- `CHART_CONFIG` — the standard `st.plotly_chart` config dict; use it on every chart

**Adding a new page:** Create `pages/<name>.py` (snake_case, no emoji) and register it in `app.py`'s `st.navigation` list with a `url_path`, then add a `ui.nav_card` for it on `home.py`. Start the page with `ui.page_header(...)`, put inputs in a `ui.panel("Parameters")` (or `st.form` if computation is expensive), import analytics from `src/` only, apply `apply_chart_theme` + `CHART_CONFIG` to all Plotly figures, and end with `ui.footer_disclaimer()`.

## Environment

Requires `.env` with `ALPHAVANTAGE_API_KEY` (copy from `.env.example`). The key is only needed for `AlphaVantageProvider`; yfinance works without any key.
