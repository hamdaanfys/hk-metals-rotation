"""Real-time rotation-onset signal.

Every value on day t uses only data up to and including day t; the
truncation tests in tests/test_lookahead.py enforce this. Hindsight labels
belong in evaluation.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalParams:
    turnover_z_window: int = 252
    z_threshold: float = 2.0
    count_window: int = 10
    min_count: int = 3
    rs_ma_window: int = 50
    onset_min_off_days: int = 20

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SignalParams":
        return cls(**config["signal"])

    def with_rule(self, z_threshold: float, min_count: int, count_window: int) -> "SignalParams":
        return replace(self, z_threshold=z_threshold, min_count=min_count, count_window=count_window)

    @property
    def label(self) -> str:
        return f"z>{self.z_threshold:g}, {self.min_count}-of-{self.count_window}"


def stock_turnover_z(close: pd.DataFrame, volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock z-score of log HKD turnover against its own trailing window.

    The window ends the day before, so today's turnover cannot set its own
    hurdle. A stock gets a z-score once it has traded for `window` sessions;
    the mean and deviation use the valid days in its trailing window and need
    at least half of them. Days with no trade or zero reported volume (Yahoo
    has scattered zero-volume days) get no z-score but do not blank the
    following window.
    """
    turnover = close.multiply(volume)
    log_turnover = np.log(turnover.where(turnover > 0))
    trailing = log_turnover.shift(1).rolling(window, min_periods=max(1, window // 2))
    sessions_traded_before = close.notna().cumsum().shift(1)
    seasoned = sessions_traded_before >= window
    return ((log_turnover - trailing.mean()) / trailing.std()).where(seasoned)


def basket_turnover_z(
    close: pd.DataFrame, volume: pd.DataFrame, weights: pd.DataFrame, window: int
) -> pd.Series:
    """Weight-averaged turnover z-score over stocks with a z-score that day.

    Averaging per-stock z-scores instead of z-scoring summed turnover keeps a
    newly listed stock from reading as a permanent turnover surge.
    """
    tickers = weights["ticker"].tolist()
    close = close[tickers].dropna(how="all")
    z = stock_turnover_z(close, volume[tickers].reindex(close.index), window)
    stock_weights = weights.set_index("ticker")["weight"]
    weight_present = z.notna().multiply(stock_weights, axis=1)
    total = weight_present.sum(axis=1)
    return z.multiply(stock_weights, axis=1).sum(axis=1).divide(total.where(total > 0))


def rotation_state(
    turnover_z: pd.Series, relative_strength: pd.Series, params: SignalParams
) -> pd.DataFrame:
    """Daily rotation state and onsets on the dates of relative_strength.

    - rotation_on: at least min_count of the last count_window days have
      turnover z > z_threshold, and relative strength is above its trailing
      rs_ma_window-day average (both windows include today).
    - signal_ready: every input to rotation_on is fully warmed up.
    - rotation_onset: the first on day after at least onset_min_off_days
      consecutive ready-and-off days.
    """
    z = turnover_z.reindex(relative_strength.index)
    hot = z > params.z_threshold
    hot_count = hot.astype(float).rolling(params.count_window, min_periods=params.count_window).sum()
    z_ready = z.notna().astype(float).rolling(params.count_window, min_periods=params.count_window).sum()
    rs_ma = relative_strength.rolling(params.rs_ma_window, min_periods=params.rs_ma_window).mean()

    ready = z_ready.eq(params.count_window) & rs_ma.notna()
    on = ready & hot_count.ge(params.min_count) & (relative_strength > rs_ma)
    off = ready & ~on
    off_before = off.astype(float).shift(1).rolling(
        params.onset_min_off_days, min_periods=params.onset_min_off_days
    ).sum()
    onset = on & off_before.eq(params.onset_min_off_days)

    return pd.DataFrame(
        {
            "turnover_z": z,
            "hot_day_count": hot_count,
            "relative_strength_ma": rs_ma,
            "signal_ready": ready,
            "rotation_on": on,
            "rotation_onset": onset,
        }
    )
