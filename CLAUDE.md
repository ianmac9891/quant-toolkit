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

**Layer separation is the core design principle.** `src/` is a pure Python library — no Streamlit imports. `pages/` is thin UI that imports from `src/`. This means analysis functions can be used from notebooks or scripts without touching Streamlit.

**Data flow:**
1. `src/data.py` defines a `Provider` ABC with `get_prices(ticker, start, end) -> DataFrame`.
2. `YFinanceProvider` is the primary implementation; `AlphaVantageProvider` is a fallback for fundamentals (25 req/day free tier — use sparingly).
3. `get_prices()` (the module-level function) wraps providers with a parquet cache in `cache/`. It loads from cache, checks if the requested date range is covered, and only hits the network for missing data. Delete files in `cache/` to force a refresh.
4. Pages call `data.get_prices()` and pass the resulting DataFrame to `src/analysis.py` functions.

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
- `fetch_sp500_tickers()` — scrapes current S&P 500 from Wikipedia; normalizes BRK.B → BRK-B
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

**`src/volforecast.py`** provides GARCH(1,1) volatility forecasting:
- `fit_garch(prices)` → `GarchFit` — fits on daily **log returns × 100** (not simple returns; log space ensures zero-drift gives flat median); raises `ValueError` if persistence ≥ 1 (non-stationary)
- `simulate_paths(fit, current_price, horizon, drift_annual, n_sim, seed)` → `VolForecast` — bootstrap resamples empirical standardized residuals (preserves fat tails); initializes GARCH variance at `h_{t+1}` (one-step-ahead), not `h_t`; returns 7 percentile paths (p2.5/10/25/50/75/90/97.5) plus `terminal_prices` array for O(1) live probability queries
- `analytic_vol_path(fit, horizon)` → `np.ndarray` — closed-form GARCH variance forecast starting from `h_{t+1}`: `h_{t+k} = h_lr + persistence^(k-1) × (h_{t+1} - h_lr)`; converges to long-run vol
- `p_above(forecast, target)` → `float` — `(terminal_prices > target).mean()`, called outside cache for live target-price updates
- `GarchFit`: stores `omega`, `alpha`, `beta`, `persistence`, `h_current_pct2`, `h_next_pct2`, `h_lr_pct2`, annualized vols, `vol_regime` ("elevated"/"normal"/"compressed"), AIC, `std_resid`
- Requires `arch>=6.3.0` (in requirements.txt)

**`src/theme.py`** is the single source of truth for colors and chart styling:
- `PRIMARY`, `BENCHMARK`, `POSITIVE`, `NEGATIVE`, `NEUTRAL` — semantic palette constants
- `PRIMARY_10/18/28/80` — opacity variants used in band fills
- `GRIDLINE`, `REFLINE` — dark-mode axis and reference-line colors
- `apply_chart_theme(fig)` — removes the white legend box, sets transparent backgrounds and dark grid lines; uses `update_xaxes`/`update_yaxes` so it works on multi-subplot figures

**Adding a new page:** Create `pages/<N>_<Name>.py` (no emoji in filename). The number controls sidebar order. Import from `src/` only. Apply `apply_chart_theme` to all Plotly figures; use semantic color constants from `src/theme.py` rather than hardcoded color strings.

## Environment

Requires `.env` with `ALPHAVANTAGE_API_KEY` (copy from `.env.example`). The key is only needed for `AlphaVantageProvider`; yfinance works without any key.
