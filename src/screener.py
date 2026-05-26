"""
Bullish stock screener — medium-horizon (6–12 month) momentum and trend signals.

Signals scored (SIGNAL_COLS, used in composite)
-----------------------------------------------
mom_12_1        : return from t-252 to t-21 (12-1 momentum; skips reversal month)
mom_6m          : trailing 126-day return
pct_above_200sma: (price / SMA200 − 1) × 100
golden_cross    : 1 if SMA50 > SMA200, else 0
dist_52w_high   : (price / 52w-high − 1) × 100  (≤ 0; less negative = closer to high)
trend_slope     : annualized OLS slope of log(price) on time, trailing 252 days
trend_r2        : R² of the same regression

Informational only (not scored)
--------------------------------
extension_z    : standard deviations the latest price is above its own OLS trendline;
                 reuses the regression already run for trend_slope/r2 — free computation.
extension_flag : categorical label based on extension_z threshold

Scoring
-------
Each SIGNAL_COLS signal is z-scored cross-sectionally with nanmean/nanstd.
Composite = equal-weight mean of z-scores.  # weights could be tuned vs forward returns

Download
--------
Each ticker is fetched individually via src.data.get_prices, which caches to
parquet so second+ runs read from disk and are fast.  Concurrent fetching with
a modest thread pool (8 workers) keeps total time reasonable.  A 2-year data
window is recommended: signals only use the trailing 252 days, so the extra
history is a buffer that lets partial downloads still clear the 202-row minimum.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TRADING_DAYS = 252
_HISTORY_MIN    = int(TRADING_DAYS * 0.8)   # 202 rows required within trailing window
_HISTORY_BUFFER = 90                        # rows required BEFORE the 252-day window
_MOM_12_1_MAX   = 5.0                       # 500%: above this is almost certainly a data artifact
_MOM_6M_MAX     = 3.0                       # 300%

SIGNAL_COLS = [
    "mom_12_1",
    "mom_6m",
    "pct_above_200sma",
    "golden_cross",
    "dist_52w_high",
    "trend_slope",
    "trend_r2",
]

_WIKI_UA = {"User-Agent": "Mozilla/5.0 (compatible; quantworkbench/1.0)"}


# ── Universe helpers ──────────────────────────────────────────────────────────

def _fetch_sp_index_tickers(url: str) -> list[str]:
    """
    Shared Wikipedia scraper.  Tries several common ticker column names.
    Uses requests + User-Agent to avoid Wikipedia's 403 on urllib's default agent.
    """
    resp = requests.get(url, headers=_WIKI_UA, timeout=15)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    for col in ("Symbol", "Ticker", "Ticker symbol"):
        if col in df.columns:
            return sorted(t.replace(".", "-") for t in df[col].dropna().tolist())
    raise ValueError(f"Could not find a ticker column in {url}. Columns: {df.columns.tolist()}")


def fetch_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia."""
    return _fetch_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )


def fetch_sp400_tickers() -> list[str]:
    """Current S&P 400 MidCap constituents from Wikipedia."""
    return _fetch_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    )


def fetch_sp600_tickers() -> list[str]:
    """Current S&P 600 SmallCap constituents from Wikipedia."""
    return _fetch_sp_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    )


def fetch_sp1500_tickers() -> list[str]:
    """
    S&P 500 + S&P 400 MidCap + S&P 600 SmallCap (deduplicated).
    Virtually all names have liquid listed options.
    Scanning ~1500 tickers takes longer than the 500-ticker preset.
    """
    combined: set[str] = set()
    for fn in (fetch_sp500_tickers, fetch_sp400_tickers, fetch_sp600_tickers):
        try:
            combined.update(fn())
        except Exception:
            pass   # partial failure — still return whatever we got
    return sorted(combined)


# ── Price download ────────────────────────────────────────────────────────────

def fetch_ticker_prices(
    ticker: str,
    start: date,
    end: date,
    max_retries: int = 3,
) -> pd.Series:
    """
    Fetch adj_close for one ticker via data.get_prices (parquet-cached).
    Retries up to max_retries times with exponential backoff on exception.
    Returns an empty Series on all-retry failure — callers treat this as a
    download failure, distinct from a ticker that returned data but < 252 rows.
    """
    from src import data   # local import avoids circular dependency at module level

    for attempt in range(max_retries):
        try:
            df = data.get_prices(ticker, start, end)
            if not df.empty and "adj_close" in df.columns:
                return df["adj_close"].rename(ticker)
            return pd.Series(dtype=float, name=ticker)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))   # 0.5 s, 1 s, 2 s

    return pd.Series(dtype=float, name=ticker)


# ── OLS (vectorized, reused for trend + extension) ────────────────────────────

def _ols_vectorized(
    log_prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized OLS: regress each column on an integer time index.

    Returns
    -------
    annualized_slope : (k,) — annualized log-return rate of the trendline
    r2               : (k,) — coefficient of determination
    extension_z      : (k,) — latest price residual / residual std dev;
                       positive = price stretched above its own trendline
    """
    n = log_prices.shape[0]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    t_ss = (t ** 2).sum()

    y_mean = np.nanmean(log_prices, axis=0)
    y_dm   = log_prices - y_mean

    slopes    = np.nansum(t[:, None] * y_dm, axis=0) / t_ss
    y_pred    = t[:, None] * slopes
    residuals = y_dm - y_pred                          # (n, k)

    ss_res = np.nansum(residuals ** 2, axis=0)
    ss_tot = np.nansum(y_dm ** 2, axis=0)

    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)

    resid_std   = np.sqrt(ss_res / max(n - 2, 1))     # (k,) — regression residual std dev
    extension_z = np.where(
        resid_std > 1e-12,
        residuals[-1, :] / resid_std,
        0.0,
    )

    return slopes * TRADING_DAYS, r2, extension_z


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SIGNAL_COLS plus extension_z / extension_flag for every ticker.
    Tickers without sufficient history in the trailing 252-day window get NaN.
    extension_z and extension_flag are informational — not used in the composite.
    """
    all_cols = SIGNAL_COLS + ["extension_z", "extension_flag"]
    nan_frame = pd.DataFrame(
        {c: float("nan") if c != "extension_flag" else None for c in all_cols},
        index=prices.columns,
    )

    if len(prices) < TRADING_DAYS:
        return nan_frame

    p = prices.ffill()

    last  = p.iloc[-1]
    p21   = p.iloc[-21]
    p126  = p.iloc[-126]
    p252  = p.iloc[-252]
    sma50  = p.rolling(50,  min_periods=50 ).mean().iloc[-1]
    sma200 = p.rolling(200, min_periods=200).mean().iloc[-1]
    high52 = p.iloc[-TRADING_DAYS:].max()

    log_window              = np.log(p.iloc[-TRADING_DAYS:].values.astype(float))
    slopes, r2_arr, ext_z   = _ols_vectorized(log_window)

    def _ext_flag(z: float) -> str:
        if np.isnan(z):
            return ""
        if z >= 2.0:
            return "Stretched"
        if z >= 1.0:
            return "Extended"
        return "On trend"

    signals = pd.DataFrame({
        "mom_12_1":         (p21  / p252   - 1.0).values,
        "mom_6m":           (last / p126   - 1.0).values,
        "pct_above_200sma": (last / sma200 - 1.0).values * 100.0,
        "golden_cross":     (sma50 > sma200).astype(float).values,
        "dist_52w_high":    (last / high52 - 1.0).values * 100.0,
        "trend_slope":      slopes,
        "trend_r2":         r2_arr,
        "extension_z":      ext_z,
        "extension_flag":   [_ext_flag(z) for z in ext_z],
    }, index=prices.columns)

    # Tickers with too few valid observations in the 252-day window
    n_valid     = (~p.iloc[-TRADING_DAYS:].isna()).sum()
    window_thin = n_valid[n_valid < _HISTORY_MIN].index

    # Tickers without enough pre-window history: the momentum reference (p.iloc[-252])
    # must not fall inside IPO/spinoff stub-price territory.  Require _HISTORY_BUFFER
    # rows of valid data beyond the 252-day window so recent spinoffs are excluded.
    n_valid_total = (~p.isna()).sum()
    shallow       = n_valid_total[n_valid_total < TRADING_DAYS + _HISTORY_BUFFER].index

    insufficient = window_thin.union(shallow)
    signals.loc[insufficient, SIGNAL_COLS + ["extension_z"]] = float("nan")
    signals.loc[insufficient, "extension_flag"]              = ""

    return signals


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_and_rank(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score SIGNAL_COLS cross-sectionally (nanmean/nanstd), then average into
    composite.  extension_z and extension_flag are preserved but not scored.
    equal weights for V1 — could be optimised against forward returns.
    """
    out    = signals.copy()
    z_cols: list[str] = []

    for col in SIGNAL_COLS:
        vals = out[col].values.astype(float)
        mu   = np.nanmean(vals)
        std  = np.nanstd(vals, ddof=1)
        z    = f"z_{col}"
        out[z] = 0.0 if std < 1e-12 else (out[col] - mu) / std
        z_cols.append(z)

    out["composite"] = out[z_cols].mean(axis=1)
    return out.sort_values("composite", ascending=False, na_position="last")


# ── Info columns ──────────────────────────────────────────────────────────────

def fetch_market_caps(tickers: list[str], max_workers: int = 20) -> pd.Series:
    """
    Market cap via fast_info.market_cap — fast, reliable, gates the filter.
    One retry on failure.
    """
    def _get(t: str) -> tuple[str, float]:
        for _ in range(2):
            try:
                mc = yf.Ticker(t).fast_info.market_cap
                return t, float(mc) if mc is not None else float("nan")
            except Exception:
                time.sleep(0.2)
        return t, float("nan")

    results: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for t, val in ex.map(_get, tickers):
            results[t] = val
    return pd.Series(results, name="market_cap")


def fetch_rev_growth(
    tickers: list[str],
    max_workers: int = 5,
    total_timeout: float = 60.0,
) -> pd.Series:
    """
    Revenue growth YoY from yf.Ticker.info — slow, flaky.
    Best-effort: NaN on failure.  Hard cap of total_timeout seconds.
    Reduced workers (5) to stay under Yahoo's concurrency limit.
    """
    def _get(t: str) -> tuple[str, float]:
        try:
            val = yf.Ticker(t).info.get("revenueGrowth")
            return t, float(val) if val is not None else float("nan")
        except Exception:
            return t, float("nan")

    results: dict[str, float] = {t: float("nan") for t in tickers}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_get, t): t for t in tickers}
        try:
            for fut in as_completed(futures, timeout=total_timeout):
                t, val = fut.result()
                results[t] = val
        except FuturesTimeout:
            pass
    return pd.Series(results, name="rev_growth_yoy")


def trailing_volatility(prices: pd.DataFrame) -> pd.Series:
    """Annualized 252-day realized vol (no extra API calls)."""
    rets = prices.ffill().pct_change().iloc[-TRADING_DAYS:]
    return rets.std(ddof=1) * np.sqrt(TRADING_DAYS)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_screen(
    prices: pd.DataFrame,
    market_caps: pd.Series,
    rev_growth: pd.Series,
    min_market_cap: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: signals → sanity-filter → score → attach info → filter.

    Returns
    -------
    ranked               : tickers with composite score, sorted descending
    insufficient_history : tickers with insufficient/shallow price history
    suspect_data         : tickers excluded for non-physical signal values (data artifacts)
    """
    signals = compute_signals(prices)

    # Detect non-physical signal values BEFORE z-scoring.  A 2000%+ momentum reading
    # inflates the cross-sectional std dev and distorts every other stock's z-score.
    # Causes: spinoff stub prices surviving the history check, unadjusted splits,
    # reused ticker symbols.  Save raw values for display, then NaN them out.
    valid_mask   = signals[SIGNAL_COLS].notna().all(axis=1)
    suspect_mask = valid_mask & (
        (signals["mom_12_1"].abs() > _MOM_12_1_MAX) |
        (signals["mom_6m"].abs()   > _MOM_6M_MAX)
    )
    suspect_data = signals.loc[suspect_mask, ["mom_12_1", "mom_6m"]].copy()
    signals.loc[suspect_mask, SIGNAL_COLS + ["extension_z"]] = float("nan")
    signals.loc[suspect_mask, "extension_flag"]              = ""

    scored = score_and_rank(signals)
    vol    = trailing_volatility(prices)

    ranked           = scored[scored["composite"].notna()].copy()
    insufficient_history = scored[
        scored["composite"].isna() & ~scored.index.isin(suspect_data.index)
    ].copy()

    ranked["trailing_vol"]   = vol.reindex(ranked.index)
    ranked["market_cap"]     = market_caps.reindex(ranked.index)
    ranked["rev_growth_yoy"] = rev_growth.reindex(ranked.index)

    if min_market_cap > 0:
        ranked = ranked[
            ranked["market_cap"].isna() | (ranked["market_cap"] >= min_market_cap)
        ]

    return ranked, insufficient_history, suspect_data
