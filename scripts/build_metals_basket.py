#!/usr/bin/env python3
"""Build a first-pass HK metals basket chart from yfinance data."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
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


def parse_args() -> argparse.Namespace:
    default_start = date.today() - timedelta(days=365 * 3 + 45)
    parser = argparse.ArgumentParser(
        description="Pull HK metals/mining names from yfinance and plot a cap-weighted basket."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", default=default_start.isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["^HSI", "^HSCE"],
        help="Yahoo benchmark tickers to compare against. Defaults to ^HSI and ^HSCE.",
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


def download_ohlcv(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No price data returned from yfinance.")

    close = data["Close"].copy()
    volume = data["Volume"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        volume = volume.to_frame(tickers[0])

    return close.sort_index(), volume.sort_index()


def make_weights(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in config["constituents"]:
        ticker = item["ticker"]
        market_cap = fetch_market_cap_hkd(ticker)
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
    tickers = weights["ticker"].tolist()
    close = close[tickers].dropna(how="all")
    first_prices = close.apply(lambda s: s.dropna().iloc[0])
    normalized = close.divide(first_prices).ffill()
    weighted = normalized.multiply(weights.set_index("ticker")["weight"], axis=1)
    return 100 * weighted.sum(axis=1)


def benchmark_index(close: pd.DataFrame, benchmark: str) -> pd.Series:
    series = close[benchmark].dropna()
    return 100 * series / series.iloc[0]


def basket_turnover_hkd(close: pd.DataFrame, volume: pd.DataFrame, tickers: list[str]) -> pd.Series:
    return close[tickers].multiply(volume[tickers]).sum(axis=1)


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

    weekly = frame["basket_index"].resample("W-FRI").last().dropna()
    weekly_return = weekly.pct_change()
    momentum_weeks = weekly_return[weekly_return > momentum_threshold].index.to_period("W-FRI")
    frame["momentum_week_gt_10pct"] = frame.index.to_period("W-FRI").isin(momentum_weeks)
    return frame[
        [
            "basket_index",
            "benchmark_index",
            "relative_strength",
            "turnover",
            "turnover_breakout",
            "momentum_week_gt_10pct",
        ]
    ]


def momentum_marker_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if not frame["momentum_week_gt_10pct"].any():
        return []

    markers = []
    flagged = frame[frame["momentum_week_gt_10pct"]].copy()
    for _, week_rows in flagged.groupby(flagged.index.to_period("W-FRI")):
        markers.append(week_rows.index[0])
    return markers


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
) -> dict[str, Any]:
    turnover_date = first_signal_date(frame, "turnover_breakout", analysis_start)
    momentum_date = first_signal_date(frame, "momentum_week_gt_10pct", analysis_start)
    lift_date = first_decisive_relative_lift(frame, analysis_start)

    row = {
        "benchmark": benchmark,
        "benchmark_label": benchmark_label(benchmark),
        "analysis_start": analysis_start,
        "first_turnover_breakout": turnover_date,
        "first_momentum_week_gt_10pct": momentum_date,
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


def plot_zijin_diagnostic(close: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    ticker = "2899.HK"
    spin_date = pd.Timestamp("2025-09-30")
    start = pd.Timestamp("2025-08-15")
    end = pd.Timestamp("2025-11-15")

    zijin = close[[ticker]].dropna().rename(columns={ticker: "adjusted_close"})
    zijin["daily_return"] = zijin["adjusted_close"].pct_change()
    window = zijin.loc[start:end].copy()

    fig, (ax_price, ax_return) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_price.plot(window.index, window["adjusted_close"], color="#1f77b4", linewidth=2)
    ax_price.axvline(spin_date, color="#b91c1c", linewidth=1.2, alpha=0.75)
    ax_price.set_title("2899.HK Zijin Mining adjusted close around Zijin Gold listing")
    ax_price.set_ylabel("Adjusted close")
    ax_price.grid(True, alpha=0.25)

    ax_return.bar(window.index, 100 * window["daily_return"], color="#8bb7d8", width=1)
    ax_return.axvline(spin_date, color="#b91c1c", linewidth=1.2, alpha=0.75)
    ax_return.axhline(0, color="#6b7280", linewidth=0.8)
    ax_return.set_ylabel("Daily return, %")
    ax_return.grid(True, alpha=0.25)
    ax_return.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_return.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return window


def plot_rotation_view(
    frame: pd.DataFrame,
    benchmark: str,
    weights: pd.DataFrame,
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
            f"Vertical red lines = weeks where basket return > 10%. "
            f"Top current-cap weights: {top_weights}."
        ),
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basket(
    basket: pd.Series,
    benchmark: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    fig, (ax_price, ax_turnover) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(basket.index, basket, label="Metals basket, cap-weighted", linewidth=2.2)
    ax_price.plot(benchmark.index, benchmark, label="Hang Seng Index", linewidth=1.6, alpha=0.8)
    ax_price.set_title(title)
    ax_price.set_ylabel("Indexed price, start = 100")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")

    rel_strength = basket.reindex(benchmark.index).ffill() - benchmark
    ax_rel = ax_price.twinx()
    ax_rel.plot(rel_strength.index, rel_strength, color="#6b7280", linewidth=1.0, alpha=0.45)
    ax_rel.set_ylabel("Relative strength vs HSI, pct pts")
    ax_rel.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.5)

    turnover_ma = turnover.rolling(20).mean()
    ax_turnover.bar(turnover.index, turnover / 1_000_000_000, color="#8bb7d8", alpha=0.55, width=1)
    ax_turnover.plot(turnover_ma.index, turnover_ma / 1_000_000_000, color="#1f77b4", linewidth=1.4)
    ax_turnover.set_ylabel("HKD turnover, bn")
    ax_turnover.grid(True, alpha=0.25)

    ax_turnover.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_turnover.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    top_weights = ", ".join(
        f"{row.ticker} {row.weight:.1%}" for row in weights.head(4).itertuples(index=False)
    )
    fig.text(
        0.01,
        0.01,
        f"Current-cap weights. Top weights: {top_weights}. Volume shown as summed daily HKD turnover.",
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
    weights = make_weights(config)
    tickers = weights["ticker"].tolist()
    benchmarks = list(dict.fromkeys(args.benchmarks or [config["benchmark"]]))

    close, volume = download_ohlcv(tickers + benchmarks, args.start, args.end)

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
            title=f"{config['name']} vs {benchmark_label(benchmark)}",
            output_path=chart_path,
        )

        rotation_summaries.append(
            {
                "benchmark": benchmark,
                "label": benchmark_label(benchmark),
                "ending_relative_strength": rotation["relative_strength"].iloc[-1],
                "turnover_breakout_days": int(rotation["turnover_breakout"].sum()),
                "momentum_weeks": int(rotation["momentum_week_gt_10pct"].groupby(rotation.index.to_period("W-FRI")).max().sum()),
                "benchmark_rows": len(rotation),
                "benchmark_missing_after_join": int(rotation["benchmark_index"].isna().sum()),
                "benchmark_max_unchanged_run": max_consecutive_unchanged(rotation["benchmark_index"]),
                "chart": chart_path,
                "csv": data_path,
            }
        )
        print(f"Wrote {chart_path}")
        print(f"Wrote {data_path}")

    timing = pd.DataFrame(
        [
            make_timing_row(
                benchmark=row["benchmark"],
                frame=pd.read_csv(row["csv"], parse_dates=["date"]).set_index("date"),
                analysis_start=args.analysis_start,
            )
            for row in rotation_summaries
        ]
    )
    timing_path = args.output_dir / "metals_rotation_timing.csv"
    timing.to_csv(timing_path, index=False)

    zijin_chart_path = args.output_dir / "zijin_2899_spin_off_check.png"
    zijin_csv_path = args.output_dir / "zijin_2899_spin_off_check.csv"
    zijin_window = plot_zijin_diagnostic(close, zijin_chart_path)
    zijin_window.to_csv(zijin_csv_path, index_label="date")

    print(f"Wrote {weights_path}")
    print(f"Wrote {timing_path}")
    print(f"Wrote {zijin_chart_path}")
    print(f"Wrote {zijin_csv_path}")
    print()
    print("Rotation summaries:")
    print(pd.DataFrame(rotation_summaries).to_string(index=False))
    print()
    print("Timing table:")
    print(timing.to_string(index=False))
    print()
    print("Weights used:")
    print(weights[["ticker", "name", "weight", "market_cap_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
