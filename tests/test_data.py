"""Offline tests for the cache-coverage logic in src/data.py.

The staleness rules are what keep a 500-name scan from re-downloading the
world every run, so they get explicit boundary coverage: weekends on both
ends of the window, the current-session exclusion, and the documented holiday
limitation. No test here touches the network; frames are built directly and
"today" is frozen by monkeypatching the date class inside the module.
"""

from datetime import date

import pandas as pd
import pytest

from src import data


class _FrozenDate(date):
    """date.today() pinned to a chosen calendar day."""
    _today = date(2026, 6, 10)   # a Wednesday

    @classmethod
    def today(cls):
        return cls._today


@pytest.fixture
def frozen(monkeypatch):
    def _freeze(d: date):
        _FrozenDate._today = d
        monkeypatch.setattr(data, "date", _FrozenDate)
    return _freeze


def _frame(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"adj_close": range(len(idx))}, index=idx)


# ── _effective_end ────────────────────────────────────────────────────────────

def test_effective_end_today_steps_back_one_session(frozen):
    # Requesting through "today" (a Wednesday) must only require the cache to
    # reach Tuesday: Wednesday's bar does not exist until the close.
    frozen(date(2026, 6, 10))
    assert data._effective_end(date(2026, 6, 10)) == pd.Timestamp("2026-06-09")


def test_effective_end_clamps_weekend_to_friday(frozen):
    # Saturday and Sunday requests are satisfied by Friday's bar.
    frozen(date(2026, 6, 10))
    assert data._effective_end(date(2026, 6, 6)) == pd.Timestamp("2026-06-05")
    assert data._effective_end(date(2026, 6, 7)) == pd.Timestamp("2026-06-05")


def test_effective_end_weekend_today(frozen):
    # When "today" is Saturday, requesting through Saturday needs Friday,
    # and Friday is complete, so no step-back applies.
    frozen(date(2026, 6, 6))
    assert data._effective_end(date(2026, 6, 6)) == pd.Timestamp("2026-06-05")


def test_effective_end_monday_today_steps_to_friday(frozen):
    # Monday before the close: the freshest complete session is Friday.
    frozen(date(2026, 6, 8))
    assert data._effective_end(date(2026, 6, 8)) == pd.Timestamp("2026-06-05")


def test_effective_end_past_business_day_is_itself(frozen):
    frozen(date(2026, 6, 10))
    assert data._effective_end(date(2026, 5, 14)) == pd.Timestamp("2026-05-14")


def test_effective_end_holiday_limitation_documented(frozen):
    # bdate_range knows weekends, not exchange holidays: July 3 2026 falls on
    # a Friday observance of Independence Day with no trading, yet it is
    # treated as a business day. The consequence is one spurious refetch per
    # holiday, after which the merged cache still ends on July 2 and the same
    # check fires again the next run. Accepted cost; this test pins the
    # current behavior so a future calendar-aware fix shows up as a diff here.
    frozen(date(2026, 7, 10))
    assert data._effective_end(date(2026, 7, 3)) == pd.Timestamp("2026-07-03")


# ── _cache_is_stale ───────────────────────────────────────────────────────────

def test_fresh_cache_covering_window_is_not_stale(frozen):
    frozen(date(2026, 6, 10))
    cached = _frame("2024-06-03", "2026-06-09")
    assert not data._cache_is_stale(cached, date(2024, 6, 3), date(2026, 6, 10))


def test_cache_ending_two_sessions_back_is_stale(frozen):
    frozen(date(2026, 6, 10))
    cached = _frame("2024-06-03", "2026-06-08")   # Monday; Tuesday is required
    assert data._cache_is_stale(cached, date(2024, 6, 3), date(2026, 6, 10))


def test_weekend_request_end_does_not_refetch(frozen):
    # End = Saturday, cache through Friday: covered.
    frozen(date(2026, 6, 10))
    cached = _frame("2024-06-03", "2026-06-05")
    assert not data._cache_is_stale(cached, date(2024, 6, 3), date(2026, 6, 6))


def test_weekend_request_start_does_not_refetch(frozen):
    # Start = Saturday 2024-06-01; the first possible bar is Monday 06-03.
    # A cache beginning Monday must count as covering the Saturday start.
    frozen(date(2026, 6, 10))
    cached = _frame("2024-06-03", "2026-06-09")
    assert not data._cache_is_stale(cached, date(2024, 6, 1), date(2026, 6, 10))


def test_cache_starting_after_requested_start_is_stale(frozen):
    frozen(date(2026, 6, 10))
    cached = _frame("2025-01-06", "2026-06-09")
    assert data._cache_is_stale(cached, date(2024, 6, 3), date(2026, 6, 10))


def test_empty_cache_is_stale(frozen):
    frozen(date(2026, 6, 10))
    assert data._cache_is_stale(pd.DataFrame(), date(2024, 6, 3), date(2026, 6, 10))
