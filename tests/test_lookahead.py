"""Truncation tests: a real-time signal on day t must not change when later data is added."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

import build_metals_basket as bmb

REAL_TIME_SIGNALS = [
    "basket_index",
    "relative_strength",
    "turnover_breakout",
    "momentum_week_above_threshold",
]
THRESHOLD = 0.10
TURNOVER_WINDOW = 60

Signals = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def rotation_signals(market) -> Signals:
    """The same pipeline main() runs, as a function of the data visible so far."""

    def compute(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        basket = bmb.normalized_cap_weighted_index(close, market.weights)
        turnover = bmb.basket_turnover_hkd(close, volume, market.weights["ticker"].tolist())
        benchmark = bmb.benchmark_index(close, market.benchmark)
        return bmb.joined_rotation_frame(basket, benchmark, turnover, TURNOVER_WINDOW, THRESHOLD)

    return compute


def truncation_mismatches(compute: Signals, market, columns: list[str]) -> list[tuple]:
    """Recompute on data cut off at each day t and compare day t against the full run."""
    full = compute(market.close, market.volume)
    mismatches = []
    for cutoff in full.index[1:]:
        truncated = compute(market.close.loc[:cutoff], market.volume.loc[:cutoff])
        for column in columns:
            seen_then = truncated.at[cutoff, column]
            seen_later = full.at[cutoff, column]
            if not (seen_then == seen_later or (pd.isna(seen_then) and pd.isna(seen_later))):
                mismatches.append((cutoff.date(), column, seen_then, seen_later))
    return mismatches


def test_synthetic_data_exercises_every_signal(market):
    full = rotation_signals(market)(market.close, market.volume)
    assert full["momentum_week_above_threshold"].sum() >= 2
    assert full["turnover_breakout"].sum() >= 5
    assert full.index[0] < market.suspension[0] and market.last_trade < full.index[-1]
    assert len(full) >= 250


def test_real_time_signals_do_not_change_when_future_data_arrives(market):
    mismatches = truncation_mismatches(rotation_signals(market), market, REAL_TIME_SIGNALS)
    assert mismatches == [], f"{len(mismatches)} look-ahead mismatches, first: {mismatches[:5]}"


def test_momentum_flag_lands_on_first_session_after_the_weeks_friday(market):
    full = rotation_signals(market)(market.close, market.volume)
    flagged = full.index[full["momentum_week_above_threshold"]]
    session_after_holiday = full.index[full.index > market.holiday_friday][0]

    # The holiday week's rally is flagged on the following Monday, not the Thursday.
    assert session_after_holiday in flagged
    assert session_after_holiday.weekday() == 0
    assert all(day.weekday() == 4 or day == session_after_holiday for day in flagged)
    # The mid-week spike reverses by Friday, so its week is never flagged.
    spike_week = full.index.to_period("W-FRI") == market.midweek_spike.to_period("W-FRI")
    assert not full.loc[spike_week, "momentum_week_above_threshold"].any()


def momentum_flag_whole_week(basket_index: pd.Series) -> pd.Series:
    """Original behavior: flag every day of a week whose Friday-to-Friday return clears the bar."""
    weekly = basket_index.resample("W-FRI").last().dropna()
    weekly_return = weekly.pct_change()
    weeks = weekly_return[weekly_return > THRESHOLD].index.to_period("W-FRI")
    return pd.Series(basket_index.index.to_period("W-FRI").isin(weeks), index=basket_index.index)


def momentum_flag_last_seen_day(basket_index: pd.Series) -> pd.Series:
    """Flag the last day of each week present in the data.

    Looks causal, but on data ending mid-week it treats the partial week as
    complete, so the full data (which knows more days follow) disagrees.
    """
    week = basket_index.index.to_period("W-FRI")
    week_close_dates = basket_index.index.to_series().groupby(week).max()
    weekly_return = basket_index.loc[week_close_dates].pct_change()
    hits = weekly_return.index[weekly_return > THRESHOLD]
    return pd.Series(basket_index.index.isin(hits), index=basket_index.index)


@pytest.mark.parametrize("lookahead_flag", [momentum_flag_whole_week, momentum_flag_last_seen_day])
def test_truncation_check_catches_lookahead_momentum(market, lookahead_flag):
    pipeline = rotation_signals(market)

    def with_lookahead_flag(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        frame = pipeline(close, volume)
        frame["momentum_week_above_threshold"] = lookahead_flag(frame["basket_index"])
        return frame

    mismatches = truncation_mismatches(
        with_lookahead_flag, market, ["momentum_week_above_threshold"]
    )
    assert mismatches, "the truncation check should flag a look-ahead momentum signal"
