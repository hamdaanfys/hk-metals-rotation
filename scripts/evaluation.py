"""Hindsight labels and evaluation of the rotation-onset signal.

Everything here may look ahead: it is for judging the real-time signals in
signals.py after the fact, never an input to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from signals import SignalParams, rotation_state


@dataclass(frozen=True)
class LiftParams:
    ma_window: int = 20
    base_window: int = 60
    min_rise: float = 0.05
    hold_sessions: int = 20
    min_gap: int = 60

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LiftParams":
        return cls(**config["lift_label"])


# Both lift-label versions are reported side by side. "revised" is a post-hoc
# revision made after seeing the data (see README) and is frozen: no further
# changes to either rule.
LIFT_LABEL_METHODS = ("original", "revised")


def lift_labels(
    relative_strength: pd.Series, params: LiftParams, method: str
) -> pd.DatetimeIndex:
    """Dates of decisive relative-strength lifts, found with hindsight.

    The "lifted" condition holds on a day when the ma_window-day average of
    relative strength is at least min_rise above the lowest relative strength
    of the prior base_window sessions. A day "holds" if the average then stays
    at or above that day's hurdle for hold_sessions sessions starting that day.
    Lifts closer than min_gap sessions to the previous lift are dropped. The
    rule is scale-free, so it means the same thing whatever level relative
    strength happens to be rebased to.

    - "original" (pre-registered): the day the condition switches on, if that
      day holds.
    - "revised" (post-hoc, frozen): in each switched-on run that began with a
      genuine switch-on, the first day that holds. Identical to "original"
      whenever the switch-on day holds; it differs only when the switch-on
      day fails the hold but the condition never switches off, which made the
      original label miss sustained rallies such as 2025-26.
      Known quirk: a sharp drop lowers the rolling base and so the hurdle, so
      the revised lift can land on a falling day at a local low (2025-02-28
      is one) rather than after a 5% rise.

    Hindsight label, not a live signal: the hold condition looks ahead by
    hold_sessions sessions by design.
    """
    if method not in LIFT_LABEL_METHODS:
        raise ValueError(f"unknown lift label method {method!r}; expected one of {LIFT_LABEL_METHODS}")
    ma = relative_strength.rolling(params.ma_window, min_periods=params.ma_window).mean()
    base = relative_strength.shift(1).rolling(params.base_window, min_periods=params.base_window).min()
    hurdle = base * (1 + params.min_rise)
    lifted = ma >= hurdle
    previous_known = ma.shift(1).notna() & hurdle.shift(1).notna()
    switches_on = lifted & previous_known & ~lifted.shift(1, fill_value=False)

    future_min = ma[::-1].rolling(params.hold_sessions, min_periods=params.hold_sessions).min()[::-1]
    holds = future_min >= hurdle

    if method == "original":
        hits = switches_on & holds
    else:
        run = (lifted != lifted.shift(1)).cumsum()
        genuine_run = switches_on.groupby(run).transform("any")
        holding = lifted & genuine_run & holds
        hits = holding & holding.astype(int).groupby(run).cumsum().eq(1)

    accepted: list[int] = []
    for position in (i for i, hit in enumerate(hits.to_numpy()) if hit):
        if not accepted or position - accepted[-1] >= params.min_gap:
            accepted.append(position)
    return relative_strength.index[accepted]


@dataclass(frozen=True)
class EvalParams:
    horizons: tuple[int, ...] = (20, 60)
    entry_lag: int = 1
    permutations: int = 10_000
    seed: int = 20260926
    lift_lead_window: int = 60
    lift_late_window: int = 20
    false_alarm_window: int = 60
    sensitivity_z: tuple[float, ...] = (1.5, 2.0, 2.5)
    sensitivity_count_rules: tuple[tuple[int, int], ...] = ((3, 10), (5, 20))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EvalParams":
        evaluation = dict(config["evaluation"])
        grid = evaluation.pop("sensitivity")
        return cls(
            horizons=tuple(evaluation.pop("horizons")),
            sensitivity_z=tuple(grid["z_threshold"]),
            sensitivity_count_rules=tuple(tuple(rule) for rule in grid["count_rule"]),
            **evaluation,
        )


def forward_relative_returns(
    relative_strength: pd.Series, horizons: tuple[int, ...], entry_lag: int
) -> pd.DataFrame:
    """Relative return from the close entry_lag sessions after day t to h sessions later.

    Relative strength is basket / benchmark, so RS[exit] / RS[entry] - 1 is the
    basket's return relative to the benchmark. NaN where the window runs past
    the data.
    """
    entry = relative_strength.shift(-entry_lag)
    return pd.DataFrame(
        {f"fwd_{h}": relative_strength.shift(-(entry_lag + h)) / entry - 1 for h in horizons}
    )


def permutation_p_values(
    pool: np.ndarray, observed: np.ndarray, permutations: int, seed: int
) -> tuple[float, float]:
    """How often len(observed) random draws from pool do at least as well.

    Returns one-sided p-values for the mean and for the hit rate (share > 0),
    each as (1 + hits) / (1 + permutations) so a p-value is never zero.
    """
    rng = np.random.default_rng(seed)
    draws = len(observed)
    random_means = np.empty(permutations)
    random_hits = np.empty(permutations)
    for i in range(permutations):
        sample = rng.choice(pool, size=draws, replace=False)
        random_means[i] = sample.mean()
        random_hits[i] = (sample > 0).mean()
    tolerance = 1e-12
    p_mean = (1 + np.sum(random_means >= observed.mean() - tolerance)) / (1 + permutations)
    p_hit = (1 + np.sum(random_hits >= (observed > 0).mean() - tolerance)) / (1 + permutations)
    return float(p_mean), float(p_hit)


def event_study(frame: pd.DataFrame, params: EvalParams) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Forward relative returns after onsets vs all signal-ready days.

    frame needs relative_strength, signal_ready and rotation_onset. The base
    rate uses every signal-ready day with a complete forward window, so it
    covers the same period the onsets could have come from.
    """
    forward = forward_relative_returns(frame["relative_strength"], params.horizons, params.entry_lag)
    events = forward[frame["rotation_onset"]].copy()
    rows = []
    for horizon in params.horizons:
        column = f"fwd_{horizon}"
        eligible = frame["signal_ready"] & forward[column].notna()
        pool = forward.loc[eligible, column].to_numpy()
        observed = forward.loc[eligible & frame["rotation_onset"], column].to_numpy()
        row: dict[str, Any] = {
            "horizon": horizon,
            "n_onsets": len(observed),
            "onset_mean": observed.mean() if len(observed) else np.nan,
            "onset_median": np.median(observed) if len(observed) else np.nan,
            "onset_hit_rate": (observed > 0).mean() if len(observed) else np.nan,
            "base_n": len(pool),
            "base_mean": pool.mean() if len(pool) else np.nan,
            "base_median": np.median(pool) if len(pool) else np.nan,
            "base_hit_rate": (pool > 0).mean() if len(pool) else np.nan,
        }
        row["excess_mean"] = row["onset_mean"] - row["base_mean"]
        if len(observed) and len(pool) > len(observed):
            row["p_mean"], row["p_hit_rate"] = permutation_p_values(
                pool, observed, params.permutations, params.seed
            )
        else:
            row["p_mean"] = row["p_hit_rate"] = np.nan
        rows.append(row)
    return events, rows


def episode_check(
    index: pd.DatetimeIndex,
    onsets: pd.DatetimeIndex,
    lifts: pd.DatetimeIndex,
    params: EvalParams,
    lift_hold_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match hindsight lifts to onsets, counting sessions along index.

    For each lift: "early" if an onset fell in the lift_lead_window sessions up
    to and including the lift (lead = sessions from the first such onset to
    the lift), "late" if the first onset came within lift_late_window sessions
    after it (negative lead), otherwise "missed".

    For each onset: "followed_by_lift" if a lift falls within
    false_alarm_window sessions (onset day included), "false_alarm" if not,
    or "pending" if the data ends before a lift in that window could have been
    labeled (a lift needs lift_hold_sessions more sessions to be confirmed).
    """
    position = pd.Series(range(len(index)), index=index)
    onset_pos = position[onsets].to_numpy()
    lift_pos = position[lifts].to_numpy()
    last = len(index) - 1

    lift_rows = []
    for lift_date, lp in zip(lifts, lift_pos):
        before = onset_pos[(onset_pos >= lp - params.lift_lead_window) & (onset_pos <= lp)]
        after = onset_pos[(onset_pos > lp) & (onset_pos <= lp + params.lift_late_window)]
        if len(before):
            status, match = "early", before.min()
        elif len(after):
            status, match = "late", after.min()
        else:
            status, match = "missed", None
        lift_rows.append(
            {
                "lift_date": lift_date,
                "status": status,
                "onset_date": index[match] if match is not None else pd.NaT,
                "lead_sessions": lp - match if match is not None else np.nan,
            }
        )

    onset_rows = []
    for onset_date, op in zip(onsets, onset_pos):
        following = lift_pos[(lift_pos >= op) & (lift_pos <= op + params.false_alarm_window)]
        if len(following):
            outcome, next_lift = "followed_by_lift", index[following.min()]
        elif op + params.false_alarm_window + lift_hold_sessions > last:
            outcome, next_lift = "pending", pd.NaT
        else:
            outcome, next_lift = "false_alarm", pd.NaT
        onset_rows.append({"onset_date": onset_date, "outcome": outcome, "next_lift": next_lift})

    lift_columns = ["lift_date", "status", "onset_date", "lead_sessions"]
    onset_columns = ["onset_date", "outcome", "next_lift"]
    return pd.DataFrame(lift_rows, columns=lift_columns), pd.DataFrame(onset_rows, columns=onset_columns)


def episode_summary(lift_episodes: pd.DataFrame, onset_outcomes: pd.DataFrame) -> dict[str, Any]:
    statuses = lift_episodes["status"].value_counts()
    outcomes = onset_outcomes["outcome"].value_counts()
    resolved = outcomes.get("followed_by_lift", 0) + outcomes.get("false_alarm", 0)
    early = lift_episodes.loc[lift_episodes["status"] == "early", "lead_sessions"]
    return {
        "n_lifts": len(lift_episodes),
        "lifts_early": int(statuses.get("early", 0)),
        "lifts_late": int(statuses.get("late", 0)),
        "lifts_missed": int(statuses.get("missed", 0)),
        "median_lead_sessions": early.median() if len(early) else np.nan,
        "onsets_followed_by_lift": int(outcomes.get("followed_by_lift", 0)),
        "false_alarms": int(outcomes.get("false_alarm", 0)),
        "onsets_pending": int(outcomes.get("pending", 0)),
        "false_alarm_rate": outcomes.get("false_alarm", 0) / resolved if resolved else np.nan,
    }


def evaluate_setting(
    frame: pd.DataFrame,
    lifts_by_label: dict[str, pd.DatetimeIndex],
    params: EvalParams,
    lift_hold_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Event study plus episode check for one signal setting, under each lift label.

    Returns the per-onset events table (with an outcome and next-lift column
    per label), the per-lift episodes table (with a lift_label column), and
    one summary row per label and horizon. The event statistics do not depend
    on the label; only the episode counts do.
    """
    forward_events, rows = event_study(frame, params)
    onsets = frame.index[frame["rotation_onset"]]
    events = frame.loc[
        onsets, ["relative_strength", "relative_strength_ma", "turnover_z", "hot_day_count"]
    ].join(forward_events)
    episodes, summary_rows = [], []
    for label, lifts in lifts_by_label.items():
        lift_episodes, onset_outcomes = episode_check(
            frame.index, onsets, lifts, params, lift_hold_sessions
        )
        events = events.join(
            onset_outcomes.set_index("onset_date").add_suffix(f"_{label}_label")
        )
        episodes.append(lift_episodes.assign(lift_label=label))
        summary = episode_summary(lift_episodes, onset_outcomes)
        summary_rows += [{"lift_label": label, **row, **summary} for row in rows]
    events.index.name = "onset_date"
    lift_episodes = pd.concat(episodes, ignore_index=True)[
        ["lift_label", "lift_date", "status", "onset_date", "lead_sessions"]
    ]
    return events, lift_episodes, summary_rows


def sensitivity_table(
    turnover_z: pd.Series,
    relative_strength: pd.Series,
    lifts_by_label: dict[str, pd.DatetimeIndex],
    signal_params: SignalParams,
    params: EvalParams,
    lift_hold_sessions: int,
) -> pd.DataFrame:
    """Rerun the evaluation over the configured z-threshold x count-rule grid.

    Only the z threshold and count rule vary; every other parameter stays at
    its configured value. The configured setting is marked is_primary so the
    table reads as a robustness check, not a menu to pick the best row from.
    """
    rows = []
    for z_threshold in params.sensitivity_z:
        for min_count, count_window in params.sensitivity_count_rules:
            setting = signal_params.with_rule(z_threshold, min_count, count_window)
            frame = rotation_state(turnover_z, relative_strength, setting).assign(
                relative_strength=relative_strength
            )
            _, _, setting_rows = evaluate_setting(frame, lifts_by_label, params, lift_hold_sessions)
            for row in setting_rows:
                rows.append(
                    {
                        "setting": setting.label,
                        "z_threshold": z_threshold,
                        "min_count": min_count,
                        "count_window": count_window,
                        "is_primary": setting == signal_params,
                        **row,
                    }
                )
    return pd.DataFrame(rows)
