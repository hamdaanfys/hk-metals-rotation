"""Shared offline fixtures: synthetic prices/volumes and a guard against yfinance calls."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

import build_metals_basket as bmb


@dataclass
class SyntheticMarket:
    close: pd.DataFrame
    volume: pd.DataFrame
    weights: pd.DataFrame
    benchmark: str
    delisted: str
    last_trade: pd.Timestamp
    suspended: str
    suspension: pd.DatetimeIndex
    holiday_friday: pd.Timestamp
    midweek_spike: pd.Timestamp


def weights_frame(weights: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame({"ticker": list(weights), "weight": list(weights.values())})
    frame["name"] = frame["ticker"]
    frame["market_cap_hkd"] = frame["weight"] * 1e11
    frame["market_cap_source"] = "synthetic"
    return frame


def make_synthetic_market(seed: int = 7, n_days: int = 280) -> SyntheticMarket:
    """Random-walk basket with the awkward cases the signals must handle.

    - DDD stops trading partway through (delisting).
    - CCC is suspended for five sessions and then resumes.
    - One Friday is a market holiday, and the Thursday before it rallies +20%.
    - A Wednesday spikes +20% and reverses by Friday, so the partial week
      clears the momentum threshold but the full week does not.
    - A Tuesday rallies +20%, so at least one full week clears the threshold.
    - Three five-session turnover surges, each with a modest rally, give the
      rotation-onset signal something to fire on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    holiday_friday = dates[dates.weekday == 4][12]
    dates = dates.drop(holiday_friday)

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    returns = rng.normal(0.0005, 0.02, size=(len(dates), len(tickers)))
    returns[dates.get_loc(holiday_friday - pd.Timedelta(days=1))] += 0.20
    midweek_spike = dates[dates.weekday == 2][30]
    spike = dates.get_loc(midweek_spike)
    returns[spike] += 0.20
    returns[spike + 2] -= 0.20
    returns[dates.get_loc(dates[dates.weekday == 1][20])] += 0.20
    surge_starts = [120, 185, 235]
    for start in surge_starts:
        returns[start : start + 5] += 0.015

    close = pd.DataFrame(10 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(
        rng.lognormal(15, 0.4, size=close.shape), index=dates, columns=tickers
    )
    volume.iloc[rng.choice(len(dates), size=15, replace=False)] *= 4
    for start in surge_starts:
        volume.iloc[start : start + 5] *= 3

    last_trade = dates[199]
    close.loc[close.index > last_trade, "DDD"] = np.nan
    suspension = dates[100:105]
    close.loc[suspension, "CCC"] = np.nan
    volume = volume.where(close.notna())

    close["BENCH"] = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=len(dates))))
    close.iloc[50, close.columns.get_loc("BENCH")] = np.nan
    volume["BENCH"] = np.nan

    return SyntheticMarket(
        close=close,
        volume=volume,
        weights=weights_frame({"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}),
        benchmark="BENCH",
        delisted="DDD",
        last_trade=last_trade,
        suspended="CCC",
        suspension=suspension,
        holiday_friday=holiday_friday,
        midweek_spike=midweek_spike,
    )


@pytest.fixture
def market() -> SyntheticMarket:
    return make_synthetic_market()


@pytest.fixture(autouse=True)
def no_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test reaches for the network."""

    def blocked(*args, **kwargs):
        raise AssertionError("tests must not call yfinance")

    monkeypatch.setattr(bmb.yf, "download", blocked)
    monkeypatch.setattr(bmb.yf, "Ticker", blocked)
