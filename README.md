# HK metals basket walking skeleton

This is the first thin slice of the rotation-detector project: one HK metals/mining basket, pulled from `yfinance`, indexed, weighted, and plotted against the Hang Seng Index.

Run:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/build_metals_basket.py
```

The script uses a local `.matplotlib` cache directory so it does not need to write into your home folder.

Outputs:

- `outputs/metals_rotation_hsi.png`: stacked rotation chart vs the Hang Seng Index.
- `outputs/metals_rotation_hsi.csv`: daily rotation data and signal flags vs the Hang Seng Index.
- `outputs/metals_rotation_hsce.png`: stacked rotation chart vs the Hang Seng China Enterprises Index.
- `outputs/metals_rotation_hsce.csv`: daily rotation data and signal flags vs the Hang Seng China Enterprises Index.
- `outputs/metals_rotation_timing.csv`: first signal dates and first decisive relative-strength lift date.
- `outputs/zijin_2899_spin_off_check.png`: adjusted-close sanity check for Zijin Mining around the Zijin Gold listing.
- `outputs/zijin_2899_spin_off_check.csv`: the Zijin adjusted close and daily return window used in that chart.
- `outputs/metals_basket_weights.csv`: the exact weights used.

The script runs both `^HSI` and `^HSCE` by default:

```bash
python scripts/build_metals_basket.py
```

You can override that if needed:

```bash
python scripts/build_metals_basket.py --benchmarks ^HSCE
```

## Weighting

The basket is current market-cap weighted. The script asks Yahoo for each constituent's latest market cap. If Yahoo does not return one, it falls back to the rough HKD market cap in `data/metals_basket.yml`; the source is written beside every name in `outputs/metals_basket_weights.csv`.

The price index is built as:

```text
constituent_normalized_price = adjusted_close_today / first_available_adjusted_close
basket_index = 100 * sum(constituent_weight * constituent_normalized_price)
```

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

Momentum is computed weekly from the basket index. Weeks with a basket return greater than 10% are marked as vertical lines on the charts, and all trading days inside those weeks are flagged in the CSV as `momentum_week_gt_10pct`.

My benchmark judgment: I would treat `^HSCE` as the primary comparison and `^HSI` as the sanity check. The basket is mostly mainland-China companies listed in Hong Kong, so HSCE is the cleaner opportunity-cost benchmark. In the current run, both benchmarks show almost the same rotation, which strengthens the read rather than weakening it.

## Timing And Data Checks

The timing table starts at `--analysis-start` (`2025-01-01` by default) and records:

- first turnover-breakout day;
- first week where basket momentum was greater than 10%;
- first decisive relative-strength lift.

The decisive-lift rule is deliberately explicit: the 20-day average of relative strength must cross above 105, the prior 60 sessions must include a base reading at or below 102, and the next 20 sessions must keep that average above 105. This is a judgment call, not a law of nature, but it is written down so it can be argued with and improved.

The script also checks benchmark sanity in the printed summary: the joined benchmark rows should have no missing values and no unchanged flatline run. For the current run, both `^HSI` and `^HSCE` pass that check.

Because Zijin Mining is the largest basket weight, the script plots `2899.HK` adjusted close around the September 30, 2025 Zijin Gold International listing. In the current Yahoo data, there is no one-day adjusted-price cliff on that date.
