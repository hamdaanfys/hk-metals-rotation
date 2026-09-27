#!/usr/bin/env python3
"""Build the HK metals basket, its rotation signals and evaluation, and the charts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import yaml

from evaluation import (
    LIFT_LABEL_METHODS,
    EvalParams,
    LiftParams,
    evaluate_setting,
    lift_labels,
    sensitivity_table,
)
from signals import SignalParams, basket_turnover_z, rotation_state


DEFAULT_CONFIG = ROOT / "data" / "metals_basket.yml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
# Pinned so reruns are reproducible. yfinance treats `end` as exclusive.
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-05-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull HK metals/mining names from yfinance and plot a cap-weighted basket."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached yfinance prices and market caps and download them again.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Yahoo benchmark tickers to compare against; the first is primary. Defaults to the "
            "config's benchmark (primary) followed by its sanity_benchmarks."
        ),
    )
    parser.add_argument(
        "--momentum-threshold",
        type=float,
        default=0.10,
        help="Weekly basket return threshold for momentum flags. Default: 0.10.",
    )
    parser.add_argument(
        "--analysis-start",
        default="2025-01-01",
        help="Start date for onset timing table. Default: 2025-01-01.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_market_cap_hkd(ticker: str) -> float | None:
    """Return current market cap in HKD when Yahoo exposes it."""
    try:
        stock = yf.Ticker(ticker)
        fast_info = stock.fast_info
        market_cap = fast_info.get("market_cap") if fast_info else None
        if market_cap:
            return float(market_cap)
    except Exception:
        pass

    try:
        info = yf.Ticker(ticker).get_info()
        market_cap = info.get("marketCap")
        if market_cap:
            return float(market_cap)
    except Exception:
        pass

    return None


def cached_market_caps(tickers: list[str], cache_dir: Path, refresh: bool = False) -> dict[str, float | None]:
    """Market caps are a live snapshot, so cache them to keep weights reproducible."""
    path = cache_dir / "market_caps.json"
    cached = {} if refresh or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    missing = [ticker for ticker in tickers if cached.get(ticker) is None]
    if missing:
        for ticker in missing:
            cached[ticker] = fetch_market_cap_hkd(ticker)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {ticker: cached.get(ticker) for ticker in tickers}


def price_cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    return cache_dir / f"{safe_name(ticker)}_{start}_{end}.csv"


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return adjusted close and volume, reading per-ticker CSVs from cache_dir when present.

    Tickers that yfinance returns no data for are left out of the result (and
    out of the cache), so callers must check which columns came back.
    """
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    to_download = []
    for ticker in tickers:
        path = price_cache_path(cache_dir, ticker, start, end)
        if path.exists() and not refresh:
            cached = pd.read_csv(path, index_col="date", parse_dates=["date"])
            closes[ticker] = cached["close"]
            volumes[ticker] = cached["volume"]
        else:
            to_download.append(ticker)

    if to_download:
        try:
            data = yf.download(
                tickers=to_download,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:
            print(f"Warning: yfinance download failed for {', '.join(to_download)}: {exc}")
            data = pd.DataFrame()

        if not data.empty:
            close = data["Close"]
            volume = data["Volume"]
            if isinstance(close, pd.Series):
                close = close.to_frame(to_download[0])
                volume = volume.to_frame(to_download[0])
            cache_dir.mkdir(parents=True, exist_ok=True)
            for ticker in to_download:
                if ticker not in close.columns or close[ticker].notna().sum() == 0:
                    continue
                frame = pd.DataFrame({"close": close[ticker], "volume": volume[ticker]}).dropna(
                    subset=["close"]
                )
                frame.to_csv(price_cache_path(cache_dir, ticker, start, end), index_label="date")
                closes[ticker] = frame["close"]
                volumes[ticker] = frame["volume"]

    if not closes:
        raise RuntimeError("No price data returned from yfinance.")

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(close.index)
    return close, volume


def make_weights(config: dict[str, Any], cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    market_caps = cached_market_caps(
        [item["ticker"] for item in config["constituents"]], cache_dir, refresh
    )
    rows = []
    for item in config["constituents"]:
        ticker = item["ticker"]
        market_cap = market_caps[ticker]
        source = "yfinance"
        if market_cap is None:
            market_cap = float(item["fallback_market_cap_hkd_bn"]) * 1_000_000_000
            source = "fallback_config"
        rows.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "market_cap_hkd": market_cap,
                "market_cap_source": source,
            }
        )

    weights = pd.DataFrame(rows)
    weights["weight"] = weights["market_cap_hkd"] / weights["market_cap_hkd"].sum()
    return weights.sort_values("weight", ascending=False).reset_index(drop=True)


def normalized_cap_weighted_index(close: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Buy-and-hold cap-weighted index that drops names on days they do not trade.

    Each day's return is taken over names with a real close that day, weighted
    by their holding value at their last known close. A delisted name simply
    never trades again, so from its first missing day the weights re-normalize
    across the survivors without a jump in the index level (it is treated as
    sold at its last price). A suspended name sits out and rejoins when it
    trades again, with the whole move since its last close landing on that day.
    Only data up to each day is used, so no future prices leak into the past.
    When every name trades every day this equals 100 * sum(weight * price / first_price).
    """
    tickers = weights["ticker"].tolist()
    close = close[tickers].dropna(how="all")
    first_prices = close.apply(lambda s: s.dropna().iloc[0])
    held = close.ffill().divide(first_prices).multiply(weights.set_index("ticker")["weight"], axis=1)
    previous = held.shift(1)
    traded = close.notna() & previous.notna()
    growth = held.where(traded).sum(axis=1) / previous.where(traded).sum(axis=1)
    growth.iloc[0] = 1.0
    return 100 * growth.cumprod()


def benchmark_index(close: pd.DataFrame, benchmark: str) -> pd.Series:
    series = close[benchmark].dropna()
    return 100 * series / series.iloc[0]


def basket_turnover_hkd(close: pd.DataFrame, volume: pd.DataFrame, tickers: list[str]) -> pd.Series:
    return close[tickers].multiply(volume[tickers]).sum(axis=1)


def resolve_benchmarks(cli_benchmarks: list[str] | None, config: dict[str, Any]) -> list[str]:
    """Return benchmarks with the primary first; --benchmarks overrides the config."""
    if cli_benchmarks:
        return list(dict.fromkeys(cli_benchmarks))
    return list(dict.fromkeys([config["benchmark"], *config.get("sanity_benchmarks", [])]))


def format_threshold(threshold: float) -> str:
    return f"{threshold * 100:g}%"


def benchmark_label(ticker: str) -> str:
    labels = {
        "^HSI": "Hang Seng Index",
        "^HSCE": "Hang Seng China Enterprises Index",
    }
    return labels.get(ticker, ticker)


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "").replace(".", "_").lower()


def joined_rotation_frame(
    basket: pd.Series,
    benchmark: pd.Series,
    turnover: pd.Series,
    turnover_z: pd.Series,
    signal_params: SignalParams,
    momentum_threshold: float,
) -> pd.DataFrame:
    frame = pd.concat(
        {
            "basket_raw": basket,
            "benchmark_raw": benchmark,
            "turnover": turnover,
        },
        axis=1,
        join="inner",
    ).dropna(subset=["basket_raw", "benchmark_raw"])

    frame["basket_index"] = 100 * frame["basket_raw"] / frame["basket_raw"].iloc[0]
    frame["benchmark_index"] = 100 * frame["benchmark_raw"] / frame["benchmark_raw"].iloc[0]
    relative_raw = frame["basket_index"] / frame["benchmark_index"]
    frame["relative_strength"] = 100 * relative_raw / relative_raw.iloc[0]
    frame = frame.join(rotation_state(turnover_z, frame["relative_strength"], signal_params))

    # A week's Friday-to-Friday return is known once its Friday has passed, so the
    # flag lands on the first trading day on or after that Friday: the Friday
    # itself, or the next session when Friday is a holiday. Using the last day
    # seen in the data instead would score partial weeks at the end of the data.
    week = frame.index.to_period("W-FRI")
    weekly_return = frame["basket_index"].groupby(week).last().pct_change()
    hit_fridays = weekly_return.index[weekly_return > momentum_threshold].end_time.normalize()
    flag_positions = frame.index.searchsorted(hit_fridays)
    momentum_dates = frame.index[flag_positions[flag_positions < len(frame)]]
    frame["momentum_week_above_threshold"] = frame.index.isin(momentum_dates)
    return frame[
        [
            "basket_index",
            "benchmark_index",
            "relative_strength",
            "relative_strength_ma",
            "turnover",
            "turnover_z",
            "hot_day_count",
            "signal_ready",
            "rotation_on",
            "rotation_onset",
            "momentum_week_above_threshold",
        ]
    ]


def max_consecutive_unchanged(series: pd.Series) -> int:
    unchanged = series.diff().abs().le(1e-12).fillna(False)
    run = 0
    max_run = 0
    for value in unchanged:
        run = run + 1 if value else 0
        max_run = max(max_run, run)
    return max_run


def first_signal_date(frame: pd.DataFrame, column: str, start: str) -> pd.Timestamp | pd.NaT:
    scoped = frame.loc[pd.Timestamp(start) :]
    hits = scoped.index[scoped[column]]
    return hits[0] if len(hits) else pd.NaT


def make_timing_row(
    benchmark: str,
    frame: pd.DataFrame,
    lifts: pd.DatetimeIndex,
    analysis_start: str,
    momentum_threshold: float,
) -> dict[str, Any]:
    onset_date = first_signal_date(frame, "rotation_onset", analysis_start)
    momentum_date = first_signal_date(frame, "momentum_week_above_threshold", analysis_start)
    later_lifts = lifts[lifts >= pd.Timestamp(analysis_start)]
    lift_date = later_lifts[0] if len(later_lifts) else pd.NaT

    row = {
        "benchmark": benchmark,
        "benchmark_label": benchmark_label(benchmark),
        "analysis_start": analysis_start,
        "momentum_threshold": momentum_threshold,
        "first_rotation_onset": onset_date,
        "first_momentum_week_above_threshold": momentum_date,
        "first_hindsight_lift_revised_label": lift_date,
    }
    for label, event_date in [
        ("onset", onset_date),
        ("momentum", momentum_date),
        ("lift", lift_date),
    ]:
        if pd.isna(event_date):
            row[f"{label}_basket_index"] = pd.NA
            row[f"{label}_benchmark_index"] = pd.NA
            row[f"{label}_relative_strength"] = pd.NA
            continue
        event = frame.loc[event_date]
        row[f"{label}_basket_index"] = event["basket_index"]
        row[f"{label}_benchmark_index"] = event["benchmark_index"]
        row[f"{label}_relative_strength"] = event["relative_strength"]
    return row


def plot_rotation_view(
    frame: pd.DataFrame,
    benchmark: str,
    weights: pd.DataFrame,
    lifts_by_label: dict[str, pd.DatetimeIndex],
    signal_params: SignalParams,
    momentum_threshold: float,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.7, 1.3]},
    )
    ax_price, ax_relative, ax_turnover = axes

    onsets = frame.index[frame["rotation_onset"]]
    for ax in axes:
        for marker in frame.index[frame["momentum_week_above_threshold"]]:
            ax.axvline(marker, color="#b91c1c", linewidth=0.7, alpha=0.12)
        for onset in onsets:
            ax.axvline(onset, color="#15803d", linewidth=1.1, alpha=0.7)

    ax_price.plot(
        frame.index,
        frame["basket_index"],
        label="Metals basket, cap-weighted",
        linewidth=1.8,
    )
    ax_price.plot(
        frame.index,
        frame["benchmark_index"],
        label=benchmark_label(benchmark),
        linewidth=1.4,
        alpha=0.85,
    )
    ax_price.set_yscale("log")
    ax_price.set_title(title)
    ax_price.set_ylabel("Rebased index (log), start = 100")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")

    ax_relative.plot(
        frame.index,
        frame["relative_strength"],
        color="#0f766e",
        linewidth=1.6,
        label="Basket / benchmark, rebased to 100",
    )
    ax_relative.plot(
        frame.index,
        frame["relative_strength_ma"],
        color="#6b7280",
        linewidth=1.0,
        alpha=0.8,
        label=f"{signal_params.rs_ma_window}-day average",
    )
    original, revised = lifts_by_label["original"], lifts_by_label["revised"]
    ax_relative.scatter(
        revised,
        frame.loc[revised, "relative_strength"],
        marker="^",
        s=70,
        color="#7c3aed",
        zorder=4,
        label=f"Hindsight lift, revised label ({len(revised)})",
    )
    ax_relative.scatter(
        original,
        frame.loc[original, "relative_strength"] * 0.9,
        marker="^",
        s=40,
        facecolors="none",
        edgecolors="#374151",
        linewidths=1.0,
        zorder=4,
        label=f"Hindsight lift, original label ({len(original)}; drawn just below)",
    )
    ax_relative.set_yscale("log")
    ax_relative.set_ylabel("Relative strength (log)")
    ax_relative.grid(True, alpha=0.25)
    ax_relative.legend(loc="upper left")

    ax_turnover.plot(
        frame.index,
        frame["turnover_z"],
        color="#1f77b4",
        linewidth=0.7,
        label="Turnover z-score (weight-averaged per stock)",
    )
    ax_turnover.axhline(
        signal_params.z_threshold,
        color="#b91c1c",
        linewidth=0.9,
        linestyle="--",
        label=f"z = {signal_params.z_threshold:g}",
    )
    ax_turnover.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.6)
    ax_turnover.set_ylabel("Turnover z")
    ax_turnover.grid(True, alpha=0.25)
    ax_turnover.legend(loc="upper left")

    ax_turnover.xaxis.set_major_locator(mdates.YearLocator())
    ax_turnover.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    top_weights = ", ".join(
        f"{row.ticker} {row.weight:.1%}" for row in weights.head(4).itertuples(index=False)
    )
    fig.text(
        0.01,
        0.01,
        (
            f"Green lines = rotation onsets ({len(onsets)}; {signal_params.label}, RS above its "
            f"{signal_params.rs_ma_window}-day average, after {signal_params.onset_min_off_days}+ off days). "
            f"Triangles = hindsight lifts (evaluation labels, not signals; revised label is post hoc).\n"
            f"Faint red lines = week-end days where the week's basket return > "
            f"{format_threshold(momentum_threshold)}. Top current-cap weights: {top_weights}."
        ),
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    signal_params = SignalParams.from_config(config)
    lift_params = LiftParams.from_config(config)
    eval_params = EvalParams.from_config(config)
    weights = make_weights(config, args.cache_dir, args.refresh_cache)
    tickers = weights["ticker"].tolist()
    benchmarks = resolve_benchmarks(args.benchmarks, config)
    primary_benchmark = benchmarks[0]

    close, volume = download_ohlcv(
        tickers + benchmarks, args.start, args.end, args.cache_dir, args.refresh_cache
    )

    usable_tickers = [ticker for ticker in tickers if ticker in close.columns and close[ticker].notna().any()]
    if len(usable_tickers) < len(tickers):
        missing = sorted(set(tickers) - set(usable_tickers))
        print(f"Warning: dropping tickers with no returned price data: {', '.join(missing)}")
        weights = weights[weights["ticker"].isin(usable_tickers)].copy()
        weights["weight"] = weights["market_cap_hkd"] / weights["market_cap_hkd"].sum()

    basket = normalized_cap_weighted_index(close, weights)
    turnover = basket_turnover_hkd(close, volume, weights["ticker"].tolist())
    turnover_z = basket_turnover_z(close, volume, weights, signal_params.turnover_z_window)

    weights_path = args.output_dir / "metals_basket_weights.csv"

    weights.to_csv(weights_path, index=False)

    rotation_summaries = []
    rotations: dict[str, tuple[pd.DataFrame, dict[str, pd.DatetimeIndex]]] = {}
    for benchmark in benchmarks:
        if benchmark not in close.columns or close[benchmark].notna().sum() == 0:
            print(f"Warning: skipping benchmark {benchmark}; no price data returned from yfinance.")
            continue
        benchmark_series = benchmark_index(close, benchmark)
        rotation = joined_rotation_frame(
            basket=basket,
            benchmark=benchmark_series,
            turnover=turnover,
            turnover_z=turnover_z,
            signal_params=signal_params,
            momentum_threshold=args.momentum_threshold,
        )
        lifts_by_label = {
            method: lift_labels(rotation["relative_strength"], lift_params, method)
            for method in LIFT_LABEL_METHODS
        }
        rotations[benchmark] = (rotation, lifts_by_label)

        suffix = safe_name(benchmark)
        chart_path = args.output_dir / f"metals_rotation_{suffix}.png"
        data_path = args.output_dir / f"metals_rotation_{suffix}.csv"
        rotation.assign(
            **{
                f"hindsight_lift_{method}": rotation.index.isin(lifts)
                for method, lifts in lifts_by_label.items()
            }
        ).to_csv(data_path, index_label="date")

        plot_rotation_view(
            frame=rotation,
            benchmark=benchmark,
            weights=weights,
            lifts_by_label=lifts_by_label,
            signal_params=signal_params,
            momentum_threshold=args.momentum_threshold,
            title=f"{config['name']} vs {benchmark_label(benchmark)}",
            output_path=chart_path,
        )

        rotation_summaries.append(
            {
                "benchmark": benchmark,
                "role": "primary" if benchmark == primary_benchmark else "sanity",
                "label": benchmark_label(benchmark),
                "ending_relative_strength": rotation["relative_strength"].iloc[-1],
                "rotation_onsets": int(rotation["rotation_onset"].sum()),
                "lifts_original_label": len(lifts_by_label["original"]),
                "lifts_revised_label": len(lifts_by_label["revised"]),
                "momentum_weeks": int(rotation["momentum_week_above_threshold"].sum()),
                "benchmark_rows": len(rotation),
                "benchmark_missing_after_join": int(rotation["benchmark_index"].isna().sum()),
                "benchmark_max_unchanged_run": max_consecutive_unchanged(rotation["benchmark_index"]),
            }
        )
        print(f"Wrote {chart_path}")
        print(f"Wrote {data_path}")

    # The timing table and the evaluation are driven by the primary benchmark;
    # the others appear only as sanity rows in the summary.
    timing_path = args.output_dir / "metals_rotation_timing.csv"
    evaluation_paths = {
        "events": args.output_dir / "rotation_onset_events.csv",
        "summary": args.output_dir / "rotation_onset_summary.csv",
        "sensitivity": args.output_dir / "rotation_onset_sensitivity.csv",
        "episodes": args.output_dir / "rotation_lift_episodes.csv",
    }
    timing = summary = sensitivity = None
    if primary_benchmark in rotations:
        primary_frame, primary_lifts = rotations[primary_benchmark]
        timing = pd.DataFrame(
            [
                make_timing_row(
                    benchmark=primary_benchmark,
                    frame=primary_frame,
                    lifts=primary_lifts["revised"],
                    analysis_start=args.analysis_start,
                    momentum_threshold=args.momentum_threshold,
                )
            ]
        )
        timing.to_csv(timing_path, index=False)

        events, lift_episodes, _ = evaluate_setting(
            primary_frame, primary_lifts, eval_params, lift_params.hold_sessions
        )
        events.to_csv(evaluation_paths["events"])
        lift_episodes.to_csv(evaluation_paths["episodes"], index=False)

        summary_rows = []
        for benchmark, (frame, lifts_by_label) in rotations.items():
            _, _, rows = evaluate_setting(frame, lifts_by_label, eval_params, lift_params.hold_sessions)
            role = "primary" if benchmark == primary_benchmark else "sanity"
            summary_rows += [
                {"benchmark": benchmark, "role": role, "setting": signal_params.label, **row}
                for row in rows
            ]
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(evaluation_paths["summary"], index=False)

        sensitivity = sensitivity_table(
            turnover_z, primary_frame["relative_strength"], primary_lifts,
            signal_params, eval_params, lift_params.hold_sessions,
        )
        sensitivity.to_csv(evaluation_paths["sensitivity"], index=False)
    else:
        for path in [timing_path, *evaluation_paths.values()]:
            path.unlink(missing_ok=True)
        print(
            f"Warning: primary benchmark {primary_benchmark} has no data; "
            "timing table and onset evaluation not written."
        )

    print(f"Wrote {weights_path}")
    if timing is not None:
        print(f"Wrote {timing_path}")
        for path in evaluation_paths.values():
            print(f"Wrote {path}")
    print()
    print(f"Primary benchmark: {primary_benchmark} ({benchmark_label(primary_benchmark)})")
    print("Rotation summaries:")
    print(pd.DataFrame(rotation_summaries).to_string(index=False))
    if timing is not None:
        print()
        print(f"Timing table (vs {primary_benchmark}):")
        print(timing.to_string(index=False))
        print()
        print(f"Onset evaluation ({signal_params.label}):")
        print(summary.to_string(index=False))
        print()
        print(f"Sensitivity (vs {primary_benchmark}):")
        print(sensitivity.to_string(index=False))
    print()
    print("Weights used:")
    print(weights[["ticker", "name", "weight", "market_cap_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
