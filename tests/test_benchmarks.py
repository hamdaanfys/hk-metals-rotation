"""A benchmark with no data is skipped instead of crashing the run."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import build_metals_basket as bmb

START, END = "2024-01-01", "2024-03-01"


def fake_yf_frame(close: pd.DataFrame) -> pd.DataFrame:
    """Shape a close frame like yf.download(group_by="column") output."""
    return pd.concat({"Close": close, "Volume": close * 0 + 1_000_000}, axis=1)


def test_download_drops_benchmark_that_returns_no_data(monkeypatch, tmp_path):
    dates = pd.bdate_range(START, periods=10)
    returned = pd.DataFrame({"AAA": np.linspace(10, 11, 10), "^DEAD": np.nan}, index=dates)
    monkeypatch.setattr(bmb.yf, "download", lambda **kwargs: fake_yf_frame(returned))

    close, volume = bmb.download_ohlcv(["AAA", "^DEAD"], START, END, tmp_path)

    assert list(close.columns) == ["AAA"]
    assert list(volume.columns) == ["AAA"]
    assert bmb.price_cache_path(tmp_path, "AAA", START, END).exists()
    # Not cached, so the next run retries it instead of trusting an empty file.
    assert not bmb.price_cache_path(tmp_path, "^DEAD", START, END).exists()


def test_download_error_keeps_cached_tickers(monkeypatch, tmp_path):
    dates = pd.bdate_range(START, periods=10)
    cached = pd.DataFrame({"close": np.linspace(10, 11, 10), "volume": 1_000_000.0}, index=dates)
    cached.to_csv(bmb.price_cache_path(tmp_path, "AAA", START, END), index_label="date")

    def download_fails(**kwargs):
        raise ConnectionError("Yahoo is down")

    monkeypatch.setattr(bmb.yf, "download", download_fails)

    close, _ = bmb.download_ohlcv(["AAA", "^DEAD"], START, END, tmp_path)

    assert list(close.columns) == ["AAA"]


def run_main(monkeypatch, tmp_path, market, benchmarks: list[str]) -> None:
    close = market.close.drop(columns=[market.benchmark])
    close["^GOOD"] = market.close[market.benchmark]
    volume = market.volume.drop(columns=[market.benchmark])

    monkeypatch.setattr(bmb, "make_weights", lambda *args, **kwargs: market.weights.copy())
    monkeypatch.setattr(bmb, "download_ohlcv", lambda *args, **kwargs: (close, volume))
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_metals_basket.py", "--output-dir", str(tmp_path), "--benchmarks", *benchmarks],
    )
    bmb.main()


def test_main_skips_sanity_benchmark_with_no_data(monkeypatch, tmp_path, market, capsys):
    run_main(monkeypatch, tmp_path, market, ["^GOOD", "^DEAD"])

    assert "skipping benchmark ^DEAD" in capsys.readouterr().out
    assert (tmp_path / "metals_rotation_good.csv").exists()
    assert (tmp_path / "metals_rotation_good.png").exists()
    assert not (tmp_path / "metals_rotation_dead.csv").exists()
    timing = pd.read_csv(tmp_path / "metals_rotation_timing.csv")
    assert timing["benchmark"].tolist() == ["^GOOD"]


def test_main_without_primary_benchmark_writes_no_timing_table(
    monkeypatch, tmp_path, market, capsys
):
    stale_timing = tmp_path / "metals_rotation_timing.csv"
    stale_timing.write_text("benchmark\n^OLD\n")

    run_main(monkeypatch, tmp_path, market, ["^DEAD", "^GOOD"])

    out = capsys.readouterr().out
    assert "primary benchmark ^DEAD has no data" in out
    assert (tmp_path / "metals_rotation_good.csv").exists()
    assert not stale_timing.exists()


@pytest.mark.parametrize("benchmarks", [["^DEAD"], ["^DEAD", "^ALSO_DEAD"]])
def test_main_with_no_usable_benchmark_does_not_crash(monkeypatch, tmp_path, market, benchmarks):
    run_main(monkeypatch, tmp_path, market, benchmarks)

    assert (tmp_path / "metals_basket_weights.csv").exists()
