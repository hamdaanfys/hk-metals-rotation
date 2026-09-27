"""Turnover z-score and rotation on/onset rules in signals.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import weights_frame
from signals import SignalParams, basket_turnover_z, rotation_state, stock_turnover_z

DATES = pd.bdate_range("2024-01-01", periods=60)


def one_stock(volume: np.ndarray, first_trade: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.DataFrame({"AAA": 10.0}, index=DATES[: len(volume)])
    close.iloc[:first_trade] = np.nan
    return close, pd.DataFrame({"AAA": volume}, index=close.index)


def test_turnover_z_uses_trailing_window_excluding_today():
    volume = np.random.default_rng(1).lognormal(10, 0.5, size=30)
    close, vol = one_stock(volume)

    z = stock_turnover_z(close, vol, window=10)["AAA"]

    log_turnover = np.log(10.0 * volume)
    day = 20
    trailing = log_turnover[day - 10 : day]
    expected = (log_turnover[day] - trailing.mean()) / trailing.std(ddof=1)
    assert z.iloc[day] == pytest.approx(expected, rel=1e-12)
    assert z.iloc[:10].isna().all() and z.iloc[10:].notna().all()


def test_zero_volume_day_does_not_blank_the_following_window():
    volume = np.random.default_rng(2).lognormal(10, 0.5, size=40)
    volume[15] = 0
    close, vol = one_stock(volume)

    z = stock_turnover_z(close, vol, window=10)["AAA"]

    assert np.isnan(z.iloc[15])
    assert z.iloc[16:].notna().all()


def test_new_listing_gets_no_z_until_it_has_traded_a_full_window():
    volume = np.random.default_rng(3).lognormal(10, 0.5, size=50)
    close, vol = one_stock(volume, first_trade=20)

    z = stock_turnover_z(close, vol, window=10)["AAA"]

    assert z.iloc[:30].isna().all()
    assert z.iloc[30:].notna().all()


def test_basket_z_reweights_over_stocks_with_a_z_score():
    rng = np.random.default_rng(4)
    close = pd.DataFrame({"AAA": 10.0, "BBB": 20.0, "CCC": 5.0}, index=DATES[:40])
    volume = pd.DataFrame(rng.lognormal(10, 0.5, size=(40, 3)), index=close.index, columns=close.columns)
    volume.iloc[30, 2] = 0  # CCC has no z-score on day 30
    weights = weights_frame({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})

    basket = basket_turnover_z(close, volume, weights, window=10)
    per_stock = stock_turnover_z(close, volume, window=10)

    day = close.index[30]
    expected = (0.5 * per_stock.at[day, "AAA"] + 0.3 * per_stock.at[day, "BBB"]) / 0.8
    assert basket[day] == pytest.approx(expected, rel=1e-12)


PARAMS = SignalParams(
    turnover_z_window=10,
    z_threshold=2.0,
    count_window=4,
    min_count=2,
    rs_ma_window=3,
    onset_min_off_days=5,
)


def state_for(hot_days: list[int], relative_strength: pd.Series | None = None) -> pd.DataFrame:
    z = pd.Series(0.0, index=DATES)
    z.iloc[hot_days] = 3.0
    if relative_strength is None:
        relative_strength = pd.Series(np.linspace(100, 160, len(DATES)), index=DATES)
    return rotation_state(z, relative_strength, PARAMS)


def test_on_needs_min_count_hot_days_in_the_count_window():
    state = state_for([20, 23])  # two hot days, 4-day window covers both on day 23 only
    assert state["rotation_on"].iloc[23]
    assert not state["rotation_on"].iloc[24]  # day 20 has left the window
    assert not state_for([20])["rotation_on"].any()


def test_on_needs_relative_strength_strictly_above_its_average():
    flat = pd.Series(100.0, index=DATES)
    assert not state_for([20, 21], flat)["rotation_on"].any()


def test_onset_needs_enough_off_days_before_it():
    state = state_for([20, 21, 26, 27, 34, 35])
    on_days = [i for i, on in enumerate(state["rotation_on"]) if on]
    onset_days = [i for i, onset in enumerate(state["rotation_onset"]) if onset]
    # On 21-23 and 27-29: the second run has only 3 off days before it, so no onset.
    # On 35-37: off 30-34 is exactly 5 days, so an onset.
    assert on_days == [21, 22, 23, 27, 28, 29, 35, 36, 37]
    assert onset_days == [21, 35]


def test_no_onset_before_the_signal_has_warmed_up():
    z = pd.Series(3.0, index=DATES)
    z.iloc[:2] = np.nan
    relative_strength = pd.Series(np.linspace(100, 160, len(DATES)), index=DATES)

    state = rotation_state(z, relative_strength, PARAMS)

    assert state["rotation_on"].iloc[10:].all()
    assert not state["rotation_onset"].any()
