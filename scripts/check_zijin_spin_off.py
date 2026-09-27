#!/usr/bin/env python3
"""Sanity-check 2899.HK adjusted close around the Zijin Gold International listing."""

from __future__ import annotations

import argparse
from pathlib import Path

# Imported first: it points MPLCONFIGDIR at the repo's .matplotlib cache.
from build_metals_basket import (
    DEFAULT_CACHE_DIR,
    DEFAULT_END,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START,
    download_ohlcv,
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TICKER = "2899.HK"
SPIN_DATE = pd.Timestamp("2025-09-30")
WINDOW_START = pd.Timestamp("2025-08-15")
WINDOW_END = pd.Timestamp("2025-11-15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def plot_zijin_diagnostic(close: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    zijin = close[[TICKER]].dropna().rename(columns={TICKER: "adjusted_close"})
    zijin["daily_return"] = zijin["adjusted_close"].pct_change()
    window = zijin.loc[WINDOW_START:WINDOW_END].copy()

    fig, (ax_price, ax_return) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_price.plot(window.index, window["adjusted_close"], color="#1f77b4", linewidth=2)
    ax_price.axvline(SPIN_DATE, color="#b91c1c", linewidth=1.2, alpha=0.75)
    ax_price.set_title("2899.HK Zijin Mining adjusted close around Zijin Gold listing")
    ax_price.set_ylabel("Adjusted close")
    ax_price.grid(True, alpha=0.25)

    ax_return.bar(window.index, 100 * window["daily_return"], color="#8bb7d8", width=1)
    ax_return.axvline(SPIN_DATE, color="#b91c1c", linewidth=1.2, alpha=0.75)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    close, _ = download_ohlcv([TICKER], args.start, args.end, args.cache_dir, args.refresh_cache)

    chart_path = args.output_dir / "zijin_2899_spin_off_check.png"
    csv_path = args.output_dir / "zijin_2899_spin_off_check.csv"
    window = plot_zijin_diagnostic(close, chart_path)
    window.to_csv(csv_path, index_label="date")

    print(f"Wrote {chart_path}")
    print(f"Wrote {csv_path}")
    largest_move = window["daily_return"].abs().max()
    print(f"Largest absolute daily return in window: {largest_move:.2%}")


if __name__ == "__main__":
    main()
