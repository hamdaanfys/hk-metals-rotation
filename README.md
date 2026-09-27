# HK metals basket walking skeleton

This is the first thin slice of the rotation-detector project: one HK metals/mining basket, pulled from `yfinance`, indexed, weighted, and plotted against the Hang Seng China Enterprises Index (`^HSCE`, primary benchmark), with the Hang Seng Index (`^HSI`) as a sanity check.

Run:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/build_metals_basket.py
python scripts/check_zijin_spin_off.py
```

The default window is pinned to `--start 2023-04-01 --end 2026-05-31` (yfinance treats `end` as exclusive) so reruns are reproducible.

Downloaded prices are cached per ticker in `data/cache/` (gitignored), keyed by ticker and date range, and the market caps used for weighting are cached in `data/cache/market_caps.json`. Reruns read from the cache; pass `--refresh-cache` to download again. Tickers that return no data are not cached, so they are retried on the next run.

The script uses a local `.matplotlib` cache directory so it does not need to write into your home folder.

Outputs:

- `outputs/metals_rotation_hsce.png`: stacked rotation chart vs the Hang Seng China Enterprises Index (primary).
- `outputs/metals_rotation_hsce.csv`: daily rotation data and signal flags vs the Hang Seng China Enterprises Index (primary).
- `outputs/metals_rotation_hsi.png`: stacked rotation chart vs the Hang Seng Index (sanity check).
- `outputs/metals_rotation_hsi.csv`: daily rotation data and signal flags vs the Hang Seng Index (sanity check).
- `outputs/metals_rotation_timing.csv`: first signal dates and first decisive relative-strength lift date vs the primary benchmark.
- `outputs/metals_basket_weights.csv`: the exact weights used.
- `outputs/zijin_2899_spin_off_check.png` / `.csv`: adjusted-close sanity check for Zijin Mining around the Zijin Gold listing, written by `scripts/check_zijin_spin_off.py`.

Benchmarks come from `data/metals_basket.yml`: `benchmark` (`^HSCE`) is the primary benchmark and drives the timing table; `sanity_benchmarks` (`^HSI`) are charted alongside it as a cross-check. By default the script runs both:

```bash
python scripts/build_metals_basket.py
```

If a benchmark download fails, that benchmark is skipped with a warning and the rest of the run continues; if the primary one fails, no timing table is written. Passing `--benchmarks` overrides the config, and the first ticker listed becomes primary:

```bash
python scripts/build_metals_basket.py --benchmarks ^HSCE
```

## Weighting

The basket is current market-cap weighted. The script asks Yahoo for each constituent's latest market cap (once; the result is cached, see above). If Yahoo does not return one, it falls back to the rough HKD market cap in `data/metals_basket.yml`; the source is written beside every name in `outputs/metals_basket_weights.csv`.

The price index is built as:

```text
constituent_normalized_price = adjusted_close_today / first_available_adjusted_close
basket_index = 100 * sum(constituent_weight * constituent_normalized_price)
```

A constituent is forward-filled only between its first and last valid price, so trading suspensions are bridged but a delisted name is not carried forward. The index is chained from daily returns over names priced on both days, weighted by current holding value, so when a name stops trading its weight is re-normalized across the remaining names from that date without a jump in the index level. With no exits this is identical to the formula above.

For this starter slice, I use current weights across the whole history. That is not perfect, but it is the right compromise for this stage: the goal is to verify the data pipeline and visually confirm the sector rally, not to produce an investable backtest. Historical weights can come later if the detector is promising.

Volume is shown as summed daily HKD turnover:

```text
basket_turnover = sum(adjusted_close * shares_traded)
```

I prefer turnover over raw share volume because one share of a high-priced miner and one share of a low-priced miner are not economically comparable.

## Rotation View

For each benchmark, the script inner-joins basket and benchmark dates, then rebases both to 100 at the first shared basket date. The relative-strength line is:

```text
relative_strength = 100 * ((basket_index / benchmark_index) / first_ratio)
```

A rising line means the metals basket is outperforming that market benchmark. That is the rotation definition used here.

Turnover breakouts use only trailing data. The threshold is yesterday's trailing turnover mean plus one trailing standard deviation, with a default 756-trading-day window. Using yesterday's trailing window makes the signal closer to what you could have known on the day instead of letting the current observation set its own hurdle.

Momentum is computed weekly (Friday-to-Friday) from the basket index. For weeks with a basket return greater than `--momentum-threshold` (default `0.10`, i.e. 10%), only the week's final trading day is flagged in the CSV as `momentum_week_above_threshold`, because that is the first day the weekly return is known. Those days are marked as vertical lines on the charts, whose caption states the threshold used; the timing table records it in `momentum_threshold`.

`^HSCE` is the primary benchmark and `^HSI` the sanity check because the basket is mostly mainland-China companies listed in Hong Kong, so HSCE is the cleaner opportunity-cost benchmark. In the current run, both benchmarks show almost the same rotation, which strengthens the read rather than weakening it.

## Timing And Data Checks

The timing table is computed against the primary benchmark, starts at `--analysis-start` (`2025-01-01` by default), and records:

- first turnover-breakout day;
- first week-end day where the week's basket return exceeded `--momentum-threshold`;
- first decisive relative-strength lift.

The decisive-lift rule is deliberately explicit: the 20-day average of relative strength must cross above 105, the prior 60 sessions must include a base reading at or below 102, and the next 20 sessions must keep that average above 105. This is a judgment call, not a law of nature, but it is written down so it can be argued with and improved.

The script also checks benchmark sanity in the printed summary: the joined benchmark rows should have no missing values and no unchanged flatline run. For the current run, both `^HSI` and `^HSCE` pass that check.

Because Zijin Mining is the largest basket weight, `scripts/check_zijin_spin_off.py` plots `2899.HK` adjusted close around the September 30, 2025 Zijin Gold International listing. In the current Yahoo data, there is no one-day adjusted-price cliff on that date.
