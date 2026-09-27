"""Forward returns, event study, permutation test, lift labels and episode matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation import (
    LIFT_LABEL_METHODS,
    EvalParams,
    LiftParams,
    episode_check,
    evaluate_setting,
    event_study,
    forward_relative_returns,
    lift_labels,
    permutation_p_values,
    sensitivity_table,
)
from signals import SignalParams

DATES = pd.bdate_range("2020-01-01", periods=300)


def test_forward_returns_start_the_session_after_the_signal():
    relative_strength = pd.Series(np.arange(1.0, 11.0), index=DATES[:10])

    forward = forward_relative_returns(relative_strength, horizons=(2,), entry_lag=1)["fwd_2"]

    # Signal on day 0 (RS 1): enter at day 1's close (RS 2), exit at day 3 (RS 4).
    assert forward.iloc[0] == pytest.approx(4 / 2 - 1)
    assert forward.iloc[6] == pytest.approx(10 / 8 - 1)
    assert forward.iloc[7:].isna().all()


def test_base_rate_uses_ready_days_with_a_complete_forward_window():
    relative_strength = pd.Series(np.linspace(100, 130, 40), index=DATES[:40])
    frame = pd.DataFrame(
        {
            "relative_strength": relative_strength,
            "signal_ready": np.arange(40) >= 10,
            "rotation_onset": np.isin(np.arange(40), [12, 20]),
        },
        index=relative_strength.index,
    )
    params = EvalParams(horizons=(5,), permutations=200)

    events, rows = event_study(frame, params)

    # Ready from day 10; a 1-day entry lag + 5-day horizon needs day t + 6 <= 39.
    assert rows[0]["base_n"] == len(range(10, 34))
    assert rows[0]["n_onsets"] == 2
    assert list(events.index) == [DATES[12], DATES[20]]


def test_permutation_p_values_are_reproducible_and_detect_planted_events():
    pool = np.random.default_rng(0).normal(0, 1, size=500)
    best, worst = np.sort(pool)[-5:], np.sort(pool)[:5]

    first_run = permutation_p_values(pool, best, 2000, seed=7)
    assert permutation_p_values(pool, best, 2000, seed=7) == first_run
    p_mean, p_hit = permutation_p_values(pool, best, 2000, seed=7)
    assert p_mean == pytest.approx(1 / 2001)
    assert p_hit < 0.1
    assert permutation_p_values(pool, worst, 2000, seed=7)[0] > 0.99


def stepped_relative_strength(scale: float = 1.0) -> pd.Series:
    """Flat at 100, then a sustained step to 112 from session 150."""
    values = np.full(len(DATES), 100.0)
    values[150:] = 112.0
    return pd.Series(values * scale, index=DATES)


@pytest.mark.parametrize("method", LIFT_LABEL_METHODS)
def test_lift_label_finds_a_sustained_rise_regardless_of_scale(method):
    params = LiftParams()

    lifts = lift_labels(stepped_relative_strength(), params, method)

    # The 20-day average first reaches 105 on the 9th session at 112 (average 105.4).
    assert list(lifts) == [DATES[158]]
    assert list(lift_labels(stepped_relative_strength(scale=37.5), params, method)) == list(lifts)


@pytest.mark.parametrize("method", LIFT_LABEL_METHODS)
def test_lift_label_needs_the_rise_to_hold(method):
    relative_strength = stepped_relative_strength()
    relative_strength.iloc[165:] = 100.0  # falls back before the 20-session hold ends

    assert len(lift_labels(relative_strength, LiftParams(), method)) == 0


def switch_on_that_fails_the_hold() -> pd.Series:
    """Switches on at session 158, dips so the hold fails, but never switches off.

    The dip drags the rolling 60-session base (and so the daily hurdle) down,
    so the condition stays on while the average falls below the switch-on
    day's hurdle. This is the pattern that hid the 2025-26 rally.
    """
    values = np.full(len(DATES), 100.0)
    values[150:164] = 112.0
    values[164:170] = 60.0
    values[170:] = 130.0
    return pd.Series(values, index=DATES)


def first_holding_day_in_run(relative_strength: pd.Series, params: LiftParams, start: int) -> int:
    """Brute-force reference for the revised rule, scanning from a switch-on day."""
    ma = relative_strength.rolling(params.ma_window).mean().to_numpy()
    base = relative_strength.shift(1).rolling(params.base_window).min().to_numpy()
    hurdle = base * (1 + params.min_rise)
    day = start
    while ma[day] >= hurdle[day]:
        if all(ma[day + k] >= hurdle[day] for k in range(params.hold_sessions)):
            return day
        day += 1
    raise AssertionError("no holding day in the run")


def test_revised_label_catches_a_run_whose_switch_on_fails_the_hold():
    relative_strength = switch_on_that_fails_the_hold()
    params = LiftParams()

    original = lift_labels(relative_strength, params, "original")
    revised = lift_labels(relative_strength, params, "revised")

    assert len(original) == 0
    assert list(revised) == [DATES[first_holding_day_in_run(relative_strength, params, 158)]]
    assert revised[0] > DATES[158]
    # Known quirk of the frozen revised rule: the dip drags the hurdle down,
    # so the lift lands inside the dip itself (session 165, RS 60).
    assert revised[0] == DATES[165]


def test_revised_label_keeps_every_original_lift_on_real_shaped_data(market):
    relative_strength = market.close["AAA"] / market.close["BENCH"].ffill()
    relative_strength = relative_strength.dropna()
    params = LiftParams(min_gap=1)

    original = set(lift_labels(relative_strength, params, "original"))
    revised = set(lift_labels(relative_strength, params, "revised"))

    assert original <= revised


def test_unknown_lift_label_method_is_rejected():
    with pytest.raises(ValueError):
        lift_labels(stepped_relative_strength(), LiftParams(), "tuned")


def test_lift_label_enforces_the_minimum_gap():
    values = np.full(len(DATES), 100.0)
    values[100:130] = 112.0  # first lift on session 108
    values[150:] = 112.0  # falls back, then a second lift on session 158, 50 later
    relative_strength = pd.Series(values, index=DATES)

    for method in LIFT_LABEL_METHODS:
        assert list(lift_labels(relative_strength, LiftParams(min_gap=60), method)) == [DATES[108]]
        assert list(lift_labels(relative_strength, LiftParams(min_gap=30), method)) == [
            DATES[108],
            DATES[158],
        ]


def test_episode_check_classifies_lifts_and_onsets():
    params = EvalParams(lift_lead_window=60, lift_late_window=20, false_alarm_window=60)
    onsets = DATES[[10, 70, 205, 250]]
    lifts = DATES[[100, 150, 200]]

    lift_episodes, onset_outcomes = episode_check(DATES, onsets, lifts, params, lift_hold_sessions=20)

    assert lift_episodes["status"].tolist() == ["early", "missed", "late"]
    assert lift_episodes["lead_sessions"].tolist()[0] == 30
    assert lift_episodes["lead_sessions"].tolist()[2] == -5
    assert np.isnan(lift_episodes["lead_sessions"].tolist()[1])
    # Onset 10: no lift within 60 sessions. 70: lift 100. 205: no lift after it
    # within 60 sessions. 250: 250 + 60 + 20 runs past the last session (299).
    assert onset_outcomes["outcome"].tolist() == [
        "false_alarm",
        "followed_by_lift",
        "false_alarm",
        "pending",
    ]


def test_sensitivity_table_covers_the_grid_and_marks_one_primary_setting():
    rng = np.random.default_rng(5)
    relative_strength = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DATES)))), index=DATES)
    turnover_z = pd.Series(rng.normal(0, 1.5, len(DATES)), index=DATES)
    signal_params = SignalParams(turnover_z_window=40, rs_ma_window=20, onset_min_off_days=10)
    params = EvalParams(permutations=100)

    no_lifts = {method: pd.DatetimeIndex([]) for method in LIFT_LABEL_METHODS}
    table = sensitivity_table(turnover_z, relative_strength, no_lifts, signal_params, params, 20)

    assert len(table) == 3 * 2 * len(LIFT_LABEL_METHODS) * len(params.horizons)
    assert table.loc[table["is_primary"], "setting"].unique().tolist() == [signal_params.label]
    assert set(table["setting"]) == {
        f"z>{z:g}, {m}-of-{w}" for z in (1.5, 2.0, 2.5) for m, w in ((3, 10), (5, 20))
    }


def test_evaluate_setting_reports_both_labels_with_identical_event_statistics():
    relative_strength = pd.Series(np.linspace(100, 160, len(DATES)), index=DATES)
    frame = pd.DataFrame(
        {
            "relative_strength": relative_strength,
            "relative_strength_ma": relative_strength,
            "turnover_z": 0.0,
            "hot_day_count": 0.0,
            "signal_ready": np.arange(len(DATES)) >= 20,
            "rotation_onset": np.isin(np.arange(len(DATES)), [50, 120]),
        },
        index=DATES,
    )
    lifts_by_label = {"original": DATES[[60]], "revised": DATES[[60, 130]]}
    params = EvalParams(permutations=100)

    events, lift_episodes, rows = evaluate_setting(frame, lifts_by_label, params, 20)

    assert events["outcome_original_label"].tolist() == ["followed_by_lift", "false_alarm"]
    assert events["outcome_revised_label"].tolist() == ["followed_by_lift", "followed_by_lift"]
    assert lift_episodes.groupby("lift_label").size().to_dict() == {"original": 1, "revised": 2}
    by_label = pd.DataFrame(rows).set_index(["lift_label", "horizon"])
    event_columns = ["n_onsets", "onset_mean", "base_mean", "p_mean", "p_hit_rate"]
    pd.testing.assert_frame_equal(
        by_label.loc["original", event_columns], by_label.loc["revised", event_columns]
    )
