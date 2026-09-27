#!/usr/bin/env python3
"""Build a first-pass HK metals basket chart from yfinance data."""

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


DEFAULT_CONFIG = ROOT / "data" / "metals_basket.yml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
# Pinned so reruns are reproducible. yfinance treats `end` as exclusive.
DEFAULT_START = "2023-04-01"
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
        "--turnover-window",
        type=int,
        default=756,
        help="Trailing trading-day window for turnover breakout threshold. Default: 756.",
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
    """Buy-and-hold cap-weighted index that drops names once they stop trading.

    Prices are forward-filled only between a ticker's first and last valid
    close, so suspensions are bridged but a delisted name is not carried
    forever. Each day's return is taken over names priced on both days,
    weighted by their current holding value, which re-normalizes weights
    across the survivors after an exit without a jump in the index level.
    With no exits this equals 100 * sum(weight * price / first_price).
    """
    tickers = weights["ticker"].tolist()
    close = close[tickers].dropna(how="all")
    alive = close.ffill().notna() & close.bfill().notna()
    close = close.ffill().where(alive)
    first_prices = close.apply(lambda s: s.dropna().iloc[0])
    held = close.divide(first_prices).multiply(weights.set_index("ticker")["weight"], axis=1)
    previous = held.shift(1)
    both = held.notna() & previous.notna()
    growth = held.where(both).sum(axis=1) / previous.where(both).sum(axis=1)
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
    turnover_window: int,
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

    window = max(1, min(turnover_window, len(frame)))
    min_periods = min(20, window)
    trailing_turnover = frame["turnover"].shift(1)
    trailing_mean = trailing_turnover.rolling(window=window, min_periods=min_periods).mean()
    trailing_std = trailing_turnover.rolling(window=window, min_periods=min_periods).std()
    frame["turnover_breakout"] = frame["turnover"] > trailing_mean + trailing_std

    # Flag only the week's final trading day: that is when the weekly return is known.
    week = frame.index.to_period("W-FRI")
    week_close_dates = frame.index.to_series().groupby(week).max()
    weekly_return = frame.loc[week_close_dates, "basket_index"].pct_change()
    momentum_dates = weekly_return.index[weekly_return > momentum_threshold]
    frame["momentum_week_above_threshold"] = frame.index.isin(momentum_dates)
    return frame[
        [
            "basket_index",
            "benchmark_index",
            "relative_strength",
            "turnover",
            "turnover_breakout",
            "momentum_week_above_threshold",
        ]
    ]


def momentum_marker_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    return frame.index[frame["momentum_week_above_threshold"]].tolist()


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


def first_decisive_relative_lift(frame: pd.DataFrame, start: str) -> pd.Timestamp | pd.NaT:
    """Find the first sustained lift off the relative-strength base.

    Judgment rule: the 20-day average of relative strength must cross above
    105, the prior 60 sessions must include a base reading at/below 102, and
    the next 20 sessions must keep the 20-day average above 105.
    """
    scoped = frame.loc[pd.Timestamp(start) :].copy()
    relative = scoped["relative_strength"]
    relative_ma = relative.rolling(20, min_periods=10).mean()
    recent_base = relative.rolling(60, min_periods=10).min().le(102)
    stays_lifted = relative_ma[::-1].rolling(20, min_periods=20).min()[::-1].gt(105)
    crosses = relative_ma.gt(105) & relative_ma.shift(1).le(105) & recent_base & stays_lifted
    hits = scoped.index[crosses.fillna(False)]
    return hits[0] if len(hits) else pd.NaT


def make_timing_row(
    benchmark: str,
    frame: pd.DataFrame,
    analysis_start: str,
    momentum_threshold: float,
) -> dict[str, Any]:
    turnover_date = first_signal_date(frame, "turnover_breakout", analysis_start)
    momentum_date = first_signal_date(frame, "momentum_week_above_threshold", analysis_start)
    lift_date = first_decisive_relative_lift(frame, analysis_start)

    row = {
        "benchmark": benchmark,
        "benchmark_label": benchmark_label(benchmark),
        "analysis_start": analysis_start,
        "momentum_threshold": momentum_threshold,
        "first_turnover_breakout": turnover_date,
        "first_momentum_week_above_threshold": momentum_date,
        "first_decisive_relative_lift": lift_date,
    }
    for label, event_date in [
        ("turnover", turnover_date),
        ("momentum", momentum_date),
        ("relative_lift", lift_date),
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
    momentum_threshold: float,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5, 1.5]},
    )
    ax_price, ax_relative, ax_turnover = axes

    marker_dates = momentum_marker_dates(frame)
    for ax in axes:
        for marker in marker_dates:
            ax.axvline(marker, color="#b91c1c", linewidth=0.9, alpha=0.28)

    ax_price.plot(
        frame.index,
        frame["basket_index"],
        label="Metals basket, cap-weighted",
        linewidth=2.2,
    )
    ax_price.plot(
        frame.index,
        frame["benchmark_index"],
        label=benchmark_label(benchmark),
        linewidth=1.7,
        alpha=0.85,
    )
    ax_price.set_title(title)
    ax_price.set_ylabel("Rebased index, start = 100")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")

    ax_relative.plot(
        frame.index,
        frame["relative_strength"],
        color="#0f766e",
        linewidth=2.0,
        label="Basket / benchmark, rebased to 100",
    )
    ax_relative.axhline(100, color="#6b7280", linewidth=0.8, alpha=0.6)
    ax_relative.set_ylabel("Relative strength")
    ax_relative.grid(True, alpha=0.25)
    ax_relative.legend(loc="upper left")

    breakout = frame[frame["turnover_breakout"]]
    ax_turnover.bar(
        frame.index,
        frame["turnover"] / 1_000_000_000,
        color="#8bb7d8",
        alpha=0.45,
        width=1,
        label="Daily turnover",
    )
    ax_turnover.scatter(
        breakout.index,
        breakout["turnover"] / 1_000_000_000,
        color="#b91c1c",
        s=18,
        label="Turnover breakout",
        zorder=3,
    )
    ax_turnover.plot(
        frame.index,
        frame["turnover"].rolling(20).mean() / 1_000_000_000,
        color="#1f77b4",
        linewidth=1.4,
        label="20-day avg",
    )
    ax_turnover.set_ylabel("HKD turnover, bn")
    ax_turnover.grid(True, alpha=0.25)
    ax_turnover.legend(loc="upper left")

    ax_turnover.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_turnover.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    top_weights = ", ".join(
        f"{row.ticker} {row.weight:.1%}" for row in weights.head(4).itertuples(index=False)
    )
    fig.text(
        0.01,
        0.01,
        (
            f"Vertical red lines = week-end days where the week's basket return > "
            f"{format_threshold(momentum_threshold)}. "
            f"Top current-cap weights: {top_weights}."
        ),
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
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

    weights_path = args.output_dir / "metals_basket_weights.csv"

    weights.to_csv(weights_path, index=False)

    rotation_summaries = []
    for benchmark in benchmarks:
        if benchmark not in close.columns or close[benchmark].notna().sum() == 0:
            print(f"Warning: skipping benchmark {benchmark}; no price data returned from yfinance.")
            continue
        benchmark_series = benchmark_index(close, benchmark)
        rotation = joined_rotation_frame(
            basket=basket,
            benchmark=benchmark_series,
            turnover=turnover,
            turnover_window=args.turnover_window,
            momentum_threshold=args.momentum_threshold,
        )

        suffix = safe_name(benchmark)
        chart_path = args.output_dir / f"metals_rotation_{suffix}.png"
        data_path = args.output_dir / f"metals_rotation_{suffix}.csv"
        rotation.to_csv(data_path, index_label="date")

        plot_rotation_view(
            frame=rotation,
            benchmark=benchmark,
            weights=weights,
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
                "turnover_breakout_days": int(rotation["turnover_breakout"].sum()),
                "momentum_weeks": int(rotation["momentum_week_above_threshold"].sum()),
                "benchmark_rows": len(rotation),
                "benchmark_missing_after_join": int(rotation["benchmark_index"].isna().sum()),
                "benchmark_max_unchanged_run": max_consecutive_unchanged(rotation["benchmark_index"]),
                "chart": chart_path,
                "csv": data_path,
            }
        )
        print(f"Wrote {chart_path}")
        print(f"Wrote {data_path}")

    # The timing table is driven by the primary benchmark only; the others are sanity checks.
    timing_path = args.output_dir / "metals_rotation_timing.csv"
    primary_rows = [row for row in rotation_summaries if row["benchmark"] == primary_benchmark]
    timing = None
    if primary_rows:
        timing = pd.DataFrame(
            [
                make_timing_row(
                    benchmark=primary_benchmark,
                    frame=pd.read_csv(primary_rows[0]["csv"], parse_dates=["date"]).set_index("date"),
                    analysis_start=args.analysis_start,
                    momentum_threshold=args.momentum_threshold,
                )
            ]
        )
        timing.to_csv(timing_path, index=False)
    else:
        timing_path.unlink(missing_ok=True)
        print(f"Warning: primary benchmark {primary_benchmark} has no data; timing table not written.")

    print(f"Wrote {weights_path}")
    if timing is not None:
        print(f"Wrote {timing_path}")
    print()
    print(f"Primary benchmark: {primary_benchmark} ({benchmark_label(primary_benchmark)})")
    print("Rotation summaries:")
    print(pd.DataFrame(rotation_summaries).to_string(index=False))
    if timing is not None:
        print()
        print(f"Timing table (vs {primary_benchmark}):")
        print(timing.to_string(index=False))
    print()
    print("Weights used:")
    print(weights[["ticker", "name", "weight", "market_cap_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
