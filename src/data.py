"""
Data layer.

Design decisions worth understanding:

1. Provider abstraction. We define a `Provider` base class with a common
   interface (`get_prices`). Concrete implementations wrap yfinance and
   Alpha Vantage. This means the rest of the codebase never imports yfinance
   or requests directly; it asks the data layer for prices and gets a clean
   DataFrame back. If yfinance breaks (it does, periodically) we can swap
   in a different provider without touching the optimizer, backtester, etc.

2. Adjusted vs raw prices. For return calculations we use adjusted close
   (handles splits and dividends). For "what was the stock trading at on
   date X" we'd use raw close. yfinance gives us both. We default to
   adjusted because every quant calculation downstream wants it.

3. Caching. Streamlit re-runs the script on every interaction. Re-pulling
   data from yfinance every time would be slow and rude. We cache to a
   parquet file per ticker. On request, we load cached data, check whether
   the requested range is already covered, and only hit the network for
   the missing portion. Parquet preserves dtypes and is ~10x faster than
   CSV for read/write.

4. Date handling. We strip timezones and work in naive UTC dates. yfinance
   returns tz-aware DatetimeIndex which causes annoying bugs when you try
   to slice it against tz-naive dates from a user input.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # pulls .env into os.environ

# Cache lives in <project_root>/cache/
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# Provider interface
# -------------------------------------------------------------------------

class Provider(ABC):
    """Common interface every data provider must implement."""

    name: str

    @abstractmethod
    def get_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Return a DataFrame indexed by date (tz-naive) with columns:
            open, high, low, close, adj_close, volume
        Empty DataFrame on failure / unknown ticker.
        """
        ...


# -------------------------------------------------------------------------
# yfinance implementation (primary)
# -------------------------------------------------------------------------

class YFinanceProvider(Provider):
    name = "yfinance"

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # yfinance's `end` is exclusive, so add a day to get inclusive behavior
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,   # we want both raw and adjusted, so don't pre-adjust
            actions=False,
        )
        if df.empty:
            return df

        # yfinance returns MultiIndex columns when given a single ticker in
        # newer versions. Flatten that.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        # Strip timezone, normalize to date-only index
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.index.name = "date"

        return df[["open", "high", "low", "close", "adj_close", "volume"]]


# -------------------------------------------------------------------------
# Alpha Vantage implementation (fallback / fundamentals)
# -------------------------------------------------------------------------

class AlphaVantageProvider(Provider):
    """
    Free tier: 25 requests/day, 5/minute. Don't loop through tickers with this.
    Useful for: fundamentals endpoints yfinance doesn't expose cleanly.
    """
    name = "alphavantage"
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY not set. Add it to .env or pass api_key="
            )

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        if "Time Series (Daily)" not in payload:
            # Common case: rate-limited or bad ticker
            return pd.DataFrame()

        raw = payload["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(raw, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
        })[["open", "high", "low", "close", "adj_close", "volume"]]
        df = df.sort_index()
        df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        return df

    def get_overview(self, ticker: str) -> dict:
        """Company overview: sector, market cap, P/E, etc. Costs 1 API call."""
        params = {"function": "OVERVIEW", "symbol": ticker, "apikey": self.api_key}
        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data or "Symbol" not in data:
            return {}
        return data


# -------------------------------------------------------------------------
# Cached access layer
# -------------------------------------------------------------------------

def _cache_path(ticker: str, provider_name: str) -> Path:
    return CACHE_DIR / f"{provider_name}_{ticker.upper().replace('/', '_')}.parquet"


def get_prices(
    ticker: str,
    start: date,
    end: date,
    provider: Optional[Provider] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Public entry point. Returns daily OHLCV + adj_close for ticker in [start, end].

    Logic:
      1. If cache exists and covers the requested range, return from cache.
      2. Else fetch missing range from provider, merge with cache, write back.
    """
    provider = provider or YFinanceProvider()
    cache_file = _cache_path(ticker, provider.name)

    cached = pd.DataFrame()
    if use_cache and cache_file.exists():
        cached = pd.read_parquet(cache_file)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    needs_fetch = (
        cached.empty
        or cached.index.min() > start_ts
        or cached.index.max() < end_ts
    )

    if needs_fetch:
        # Fetch a generous range (the full window) and merge.
        # Could be smarter and only fetch the gap, but simpler is fine here.
        fetched = provider.get_prices(ticker, start, end)
        if fetched.empty and cached.empty:
            return fetched  # nothing we can do
        if not fetched.empty:
            combined = pd.concat([cached, fetched])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(cache_file)
            cached = combined

    # Slice to requested range
    return cached.loc[start_ts:end_ts].copy()


def clear_cache(ticker: Optional[str] = None) -> int:
    """Delete cached parquet files. If ticker is None, clear everything. Returns count deleted."""
    count = 0
    for f in CACHE_DIR.glob("*.parquet"):
        if ticker is None or ticker.upper() in f.name:
            f.unlink()
            count += 1
    return count
