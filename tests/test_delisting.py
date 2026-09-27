"""Delisted and suspended constituents in normalized_cap_weighted_index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import build_metals_basket as bmb
from conftest import weights_frame


def test_delisted_stock_is_not_forward_filled():
    # AAA's last print is day 3; BBB and CCC keep trading.
    dates = pd.bdate_range("2024-01-01", periods=7)
    close = pd.DataFrame(
        {
            "AAA": [10, 11, 12, 13, np.nan, np.nan, np.nan],
            "BBB": [10, 10, 10, 10, 11, 12, 12.6],
            "CCC": [20, 20, 21, 20, 20, 19, 19.0],
        },
        index=dates,
    )
    weights = weights_frame({"AAA": 0.5, "BBB": 0.25, "CCC": 0.25})

    basket = bmb.normalized_cap_weighted_index(close, weights)
    survivors = bmb.normalized_cap_weighted_index(
        close[["BBB", "CCC"]], weights_frame({"BBB": 0.25, "CCC": 0.25})
    )

    after_exit = dates[4:]
    np.testing.assert_allclose(
        basket.pct_change().loc[after_exit],
        survivors.pct_change().loc[after_exit],
        rtol=1e-12,
        atol=1e-12,
    )
    # Forward-filling AAA at 13 would hold its half of the basket flat and
    # dilute day 4's move: BBB's +10% would show up as about +2%, not +5%.
    assert basket.pct_change().iloc[4] == pytest.approx(0.05)


def test_no_index_jump_on_the_day_a_stock_stops_trading():
    # AAA doubles, then stops trading after day 2; survivors are flat afterwards.
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame(
        {
            "AAA": [10, 15, 20, np.nan, np.nan],
            "BBB": [10, 10, 10, 10, 10.0],
            "CCC": [20, 20, 20, 20, 20.0],
        },
        index=dates,
    )
    weights = weights_frame({"AAA": 0.5, "BBB": 0.25, "CCC": 0.25})

    basket = bmb.normalized_cap_weighted_index(close, weights)

    last_trade, first_missing = dates[2], dates[3]
    assert basket[first_missing] == basket[last_trade]
    assert basket[dates[4]] == basket[last_trade]
    # Guard against a vacuous pass: naively re-weighting the survivors' own
    # price relatives (both 1.0) would drop the index from 150 to 100.
    assert basket[last_trade] == pytest.approx(150.0)


def test_synthetic_delisting_matches_survivor_only_basket(market):
    close = market.close[market.weights["ticker"]]
    basket = bmb.normalized_cap_weighted_index(close, market.weights)
    survivors = market.weights[market.weights["ticker"] != market.delisted]
    survivor_basket = bmb.normalized_cap_weighted_index(close[survivors["ticker"]], survivors)

    after = close.index[close.index > market.last_trade]
    assert close.loc[after, market.delisted].isna().all()
    np.testing.assert_allclose(
        basket.pct_change().loc[after],
        survivor_basket.pct_change().loc[after],
        rtol=1e-12,
        atol=1e-12,
    )


def test_suspended_stock_rejoins_and_its_gap_lands_on_one_day():
    # BBB is suspended on days 2-3 and resumes 20% above its last price.
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.DataFrame(
        {
            "AAA": [10, 10, 10, 10, 10, 10.0],
            "BBB": [10, 10, np.nan, np.nan, 12, 12.0],
        },
        index=dates,
    )
    weights = weights_frame({"AAA": 0.5, "BBB": 0.5})

    basket = bmb.normalized_cap_weighted_index(close, weights)

    assert basket[dates[2]] == basket[dates[1]] == basket[dates[3]] == 100.0
    assert basket[dates[4]] == pytest.approx(110.0)  # the whole +20% on BBB's half, on one day
