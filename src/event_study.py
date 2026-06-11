"""
Event study — market-model abnormal returns around a user-specified event date.

Method (Brown & Warner 1985 OLS):
  Estimation window: trading days [-est_days-buffer, -buffer) before event
  Event window:      trading days [-pre, +post] around event
  AR_t = R_i,t - (alpha + beta * R_m,t)
  CAR  = sum(AR_t) over event window
  SE(CAR) = sigma_e * sqrt(L),  L = event window length
  t-stat = CAR / SE(CAR),  df = N - 2

Additional test statistics:
  Patell (1976): standardized abnormal returns SAR_t = AR_t / (sigma_e * sqrt(C_t)),
    where C_t = 1 + 1/n + (R_mt - R̄_m)^2 / SS_m corrects for out-of-sample
    prediction error. Z = sum(SAR_t) / sqrt(L * (n-2)/(n-4)), asymptotically N(0,1).
  BMP / Boehmer-Musumeci-Poulsen (1991), multi-event only: per-event standardized
    CARs (Patell scaling), then a cross-sectional t-test on the SCARs. Robust to
    event-induced variance inflation, which both the naive and Patell tests assume away.

Multi-event: cross-sectional average of CARs, t-stat across the distribution,
plus aggregated Patell Z and the BMP t-statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src import data as dt


@dataclass
class MarketModelFit:
    alpha: float
    beta: float
    r_squared: float
    sigma_e: float   # residual std error, daily
    n_obs: int       # estimation window observations used
    mkt_mean: float  # mean market return in the estimation window (Patell C_t)
    mkt_ss: float    # sum of squared market-return deviations (Patell C_t)


@dataclass
class EventResult:
    event_date: date
    ticker: str
    benchmark: str
    event_times: np.ndarray      # integer day offsets, e.g. -5..+5
    ar: np.ndarray               # abnormal return per day
    car: np.ndarray              # cumulative AR (running sum)
    actual_return: np.ndarray
    predicted_return: np.ndarray
    fit: MarketModelFit
    car_total: float
    se_car: float
    t_stat: float                # Brown-Warner naive t
    p_value: float
    significant: bool            # two-sided, alpha=0.05, on the naive t
    patell_z: float = float("nan")
    patell_p: float = float("nan")
    scar: float = float("nan")   # standardized CAR (Patell scaling) — BMP input


@dataclass
class MultiEventResult:
    ticker: str
    benchmark: str
    per_event: List[EventResult]
    event_times: np.ndarray
    mean_ar: np.ndarray
    mean_car: np.ndarray
    se_mean_car: float
    t_stat: float                # cross-sectional t on raw CARs
    p_value: float
    significant: bool
    patell_z: float = float("nan")
    patell_p: float = float("nan")
    bmp_t: float = float("nan")
    bmp_p: float = float("nan")


# ── Data loading ──────────────────────────────────────────────────────────────

def _fetch_returns(
    ticker: str,
    benchmark: str,
    event_dates: List[date],
    estimation_days: int,
    buffer_days: int,
    pre_event: int,
    post_event: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return aligned daily simple-return series covering all events."""
    earliest = min(event_dates)
    latest   = max(event_dates)

    # Fetch start: enough trading days before earliest event for estimation + buffer
    fetch_start = pd.bdate_range(
        end=earliest, periods=estimation_days + buffer_days + pre_event + 60
    )[0].date()
    # Fetch end: enough trading days after latest event for event window
    fetch_end = pd.bdate_range(
        start=latest, periods=post_event + 15
    )[-1].date()

    df_t = dt.get_prices(ticker,    fetch_start, fetch_end)
    df_m = dt.get_prices(benchmark, fetch_start, fetch_end)

    if df_t.empty or "adj_close" not in df_t.columns:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    if df_m.empty or "adj_close" not in df_m.columns:
        raise ValueError(f"No price data returned for benchmark '{benchmark}'.")

    ret_t = df_t["adj_close"].dropna().pct_change().dropna()
    ret_m = df_m["adj_close"].dropna().pct_change().dropna()

    common = ret_t.index.intersection(ret_m.index)
    return ret_t.loc[common], ret_m.loc[common]


# ── Market model OLS ──────────────────────────────────────────────────────────

def _fit_market_model(
    ret_t: pd.Series,
    ret_m: pd.Series,
    event_date: date,
    estimation_days: int,
    buffer_days: int,
) -> MarketModelFit:
    """OLS on the estimation window ending buffer_days before the event."""
    idx = ret_t.index

    # Latest date in estimation window: last trading day at least buffer_days before event
    before_event = idx[idx < pd.Timestamp(event_date)]
    if len(before_event) < buffer_days:
        raise ValueError(
            f"Fewer than {buffer_days} trading days found before {event_date}. "
            "Extend the date range or reduce the buffer."
        )
    est_end = before_event[-buffer_days]

    # Start of estimation window: go back estimation_days from est_end
    up_to_end = idx[idx <= est_end]
    if len(up_to_end) < estimation_days:
        raise ValueError(
            f"Only {len(up_to_end)} trading days available for the estimation window "
            f"(need {estimation_days}). Extend the start date."
        )
    est_start = up_to_end[-estimation_days]

    mask = (idx >= est_start) & (idx <= est_end)
    y = ret_t.loc[mask].values
    x = ret_m.loc[mask].values
    n = len(y)

    X = np.column_stack([np.ones(n), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    resid  = y - (X @ coeffs)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_sq   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    sigma_e = float(np.sqrt(ss_res / max(n - 2, 1)))

    return MarketModelFit(
        alpha=alpha, beta=beta, r_squared=r_sq, sigma_e=sigma_e, n_obs=n,
        mkt_mean=float(x.mean()), mkt_ss=float(((x - x.mean()) ** 2).sum()),
    )


# ── Event-window extraction ───────────────────────────────────────────────────

def _event_window(
    ret_t: pd.Series,
    ret_m: pd.Series,
    event_date: date,
    fit: MarketModelFit,
    pre_event: int,
    post_event: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (event_times, ar, car, actual, predicted) for one event."""
    idx = ret_t.index

    # Snap to nearest trading day at or after event date
    candidates = idx[idx >= pd.Timestamp(event_date)]
    if len(candidates) == 0:
        raise ValueError(f"Event date {event_date} is beyond available price data.")
    event_ts   = candidates[0]
    event_pos  = idx.get_loc(event_ts)

    positions = [p for p in range(event_pos - pre_event, event_pos + post_event + 1)
                 if 0 <= p < len(idx)]

    event_times  = np.array([p - event_pos for p in positions])
    actual_ret   = ret_t.iloc[positions].values
    market_ret   = ret_m.iloc[positions].values
    predicted    = fit.alpha + fit.beta * market_ret
    ar           = actual_ret - predicted
    car          = np.cumsum(ar)

    return event_times, ar, car, actual_ret, predicted, market_ret


def _patell_stats(
    fit: MarketModelFit, ar: np.ndarray, market_ret: np.ndarray
) -> Tuple[float, float, float]:
    """(patell_z, patell_p, scar) for one event window.

    C_t corrects each prediction for estimation error; SAR variance under the
    market model is (n−2)/(n−4). SCAR is the Patell-scaled standardized CAR
    used as the input to the BMP cross-sectional test.
    """
    n, L = fit.n_obs, len(ar)
    if L == 0 or n <= 4 or fit.sigma_e <= 0 or fit.mkt_ss <= 0:
        return float("nan"), float("nan"), float("nan")

    c_t = 1.0 + 1.0 / n + (market_ret - fit.mkt_mean) ** 2 / fit.mkt_ss
    sar = ar / (fit.sigma_e * np.sqrt(c_t))

    z = float(sar.sum() / np.sqrt(L * (n - 2) / (n - 4)))
    p = float(2 * stats.norm.sf(abs(z)))
    scar = float(ar.sum() / (fit.sigma_e * np.sqrt(c_t.sum())))
    return z, p, scar


# ── Public API ────────────────────────────────────────────────────────────────

def run_single_event(
    ticker: str,
    event_date: date,
    benchmark: str = "SPY",
    estimation_days: int = 250,
    buffer_days: int = 30,
    pre_event: int = 5,
    post_event: int = 5,
) -> EventResult:
    """Run a single-event market-model event study."""
    ret_t, ret_m = _fetch_returns(
        ticker, benchmark, [event_date], estimation_days, buffer_days, pre_event, post_event
    )
    fit = _fit_market_model(ret_t, ret_m, event_date, estimation_days, buffer_days)
    event_times, ar, car, actual_ret, predicted, market_ret = _event_window(
        ret_t, ret_m, event_date, fit, pre_event, post_event
    )

    L         = len(ar)
    car_total = float(car[-1]) if L > 0 else 0.0
    se_car    = fit.sigma_e * np.sqrt(L)
    t_stat    = float(car_total / se_car) if se_car > 0 else np.nan
    df        = fit.n_obs - 2
    p_value   = float(2 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan
    patell_z, patell_p, scar = _patell_stats(fit, ar, market_ret)

    return EventResult(
        event_date=event_date, ticker=ticker, benchmark=benchmark,
        event_times=event_times, ar=ar, car=car,
        actual_return=actual_ret, predicted_return=predicted,
        fit=fit, car_total=car_total, se_car=se_car,
        t_stat=t_stat, p_value=p_value,
        significant=bool(p_value < 0.05) if np.isfinite(p_value) else False,
        patell_z=patell_z, patell_p=patell_p, scar=scar,
    )


def run_multi_event(
    ticker: str,
    event_dates: List[date],
    benchmark: str = "SPY",
    estimation_days: int = 250,
    buffer_days: int = 30,
    pre_event: int = 5,
    post_event: int = 5,
) -> MultiEventResult:
    """Run an event study across multiple dates and aggregate cross-sectionally."""
    ret_t, ret_m = _fetch_returns(
        ticker, benchmark, event_dates, estimation_days, buffer_days, pre_event, post_event
    )

    per_event: List[EventResult] = []
    for ed in event_dates:
        try:
            fit = _fit_market_model(ret_t, ret_m, ed, estimation_days, buffer_days)
            event_times, ar, car, actual_ret, pred, mkt = _event_window(
                ret_t, ret_m, ed, fit, pre_event, post_event
            )
            L        = len(ar)
            car_tot  = float(car[-1]) if L > 0 else 0.0
            se_c     = fit.sigma_e * np.sqrt(L)
            t_ev     = float(car_tot / se_c) if se_c > 0 else np.nan
            df       = fit.n_obs - 2
            p_ev     = float(2 * stats.t.sf(abs(t_ev), df=df)) if np.isfinite(t_ev) else np.nan
            pz, pp, scar = _patell_stats(fit, ar, mkt)
            per_event.append(EventResult(
                event_date=ed, ticker=ticker, benchmark=benchmark,
                event_times=event_times, ar=ar, car=car,
                actual_return=actual_ret, predicted_return=pred,
                fit=fit, car_total=car_tot, se_car=se_c,
                t_stat=t_ev, p_value=p_ev,
                significant=bool(p_ev < 0.05) if np.isfinite(p_ev) else False,
                patell_z=pz, patell_p=pp, scar=scar,
            ))
        except (ValueError, IndexError):
            continue

    if not per_event:
        raise ValueError("No events had sufficient data to process.")

    min_len     = min(len(e.ar) for e in per_event)
    event_times = per_event[0].event_times[:min_len]
    ar_matrix   = np.array([e.ar[:min_len] for e in per_event])
    mean_ar     = ar_matrix.mean(axis=0)
    mean_car    = np.cumsum(mean_ar)

    cars = np.array([e.car_total for e in per_event])
    n    = len(cars)
    se_cs = float(cars.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    t_cs  = float(cars.mean() / se_cs) if np.isfinite(se_cs) and se_cs > 0 else np.nan
    p_cs  = float(2 * stats.t.sf(abs(t_cs), df=n - 1)) if np.isfinite(t_cs) else np.nan

    # Aggregated Patell Z: sum of per-event Z components, scaled by sqrt(N)
    zs = np.array([e.patell_z for e in per_event])
    zs = zs[np.isfinite(zs)]
    if len(zs) > 0:
        patell_z = float(zs.sum() / np.sqrt(len(zs)))
        patell_p = float(2 * stats.norm.sf(abs(patell_z)))
    else:
        patell_z, patell_p = np.nan, np.nan

    # BMP: cross-sectional t-test on the Patell-scaled SCARs — robust to
    # event-induced variance inflation.
    scars = np.array([e.scar for e in per_event])
    scars = scars[np.isfinite(scars)]
    if len(scars) > 1 and scars.std(ddof=1) > 0:
        bmp_t = float(scars.mean() / (scars.std(ddof=1) / np.sqrt(len(scars))))
        bmp_p = float(2 * stats.t.sf(abs(bmp_t), df=len(scars) - 1))
    else:
        bmp_t, bmp_p = np.nan, np.nan

    return MultiEventResult(
        ticker=ticker, benchmark=benchmark,
        per_event=per_event, event_times=event_times,
        mean_ar=mean_ar, mean_car=mean_car,
        se_mean_car=se_cs, t_stat=t_cs, p_value=p_cs,
        significant=bool(p_cs < 0.05) if np.isfinite(p_cs) else False,
        patell_z=patell_z, patell_p=patell_p, bmp_t=bmp_t, bmp_p=bmp_p,
    )
