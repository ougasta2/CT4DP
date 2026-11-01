# Ten-year study inputs: 2016–2025

Source: **Energinet Energi Data Service, CC BY 4.0**. This directory contains the
selected source fields used by the ten-year study, not the entire upstream catalog.
All original JSON response envelopes and metadata are in `../cache/`. Consolidated
files are derived exports; no endorsement by Energinet is implied.

| File | Contents |
| --- | --- |
| `production_2016_2025.csv.gz` | Original selected-field production/consumption values for all ten years, keyed by UTC hour and price area |
| `prices_hourly_2016_2025.csv.gz` | Original hourly prices through September 2025, plus the complete-quarter hourly bridge for October–December 2025 |
| `manifest.json` | SHA-256 checksums, source URLs, request bounds and download times for original responses and derived artifacts |
| `../../results/decade/processed.csv.gz` | Full harmonized features, raw source solar values, repair flags, fixed price events and K-means labels |
| `../../results/decade/holdout_predictions.csv.gz` | Both held-out years with cluster labels, class probabilities and regression estimates |

Each input export contains **175,344 area-hour rows**: 140,256 training rows in
2016–2023 and 35,088 test rows in 2024–2025. These are completed Copenhagen calendar
years; use UTC keys rather than local wall-clock strings for joins and DST.
The first UTC interval is 2015-12-31 23:00; end is exclusive at the start of 2026
local time. Both DK1 and DK2 are covered. `coverage.csv` and `usable_coverage.csv`
in results/decade audit every month/year/area against a complete hourly grid.

## Explicit historical bridges

- Before 2020, all solar is stored in `SolarPowerSelfConMWh` under the source's
  legacy layout. It cannot be interpreted as measured self-consumption alone.
  From 2020, the solar proxy sums the three grid components and estimated self-use.
- 2020 contains 1,747 negative estimated self-use values. The official metadata
  defines this estimate as zero when modeled-minus-actual production is negative.
  The harmonized feature applies that bound, **retains the signed source values**,
  and flags each affected row in `solar_estimate_floor_applied`. This is an explicit
  historical repair, not an imputation of missing values. The separate strict
  no-repair and no-solar clustering sensitivities disclose its influence.
- From October 1, 2025, `DayAheadPrices` provides four 15-minute EUR/MWh prices
  per UTC hour. Only complete, unique sets of four are averaged. An hourly event
  is defined on that mean, not on the existence of a negative quarter; the export
  retains the quarter-negative fraction for comparison.

Original files are unchanged. The production export retains original signed
values; corrections appear only in explicitly named harmonized analysis columns.
No missing observation is turned into a zero.

## Reproduce

From the repository root after installing dependencies:

```powershell
.venv/Scripts/python scripts/run_decade.py --offline
.venv/Scripts/python scripts/execute_notebook.py notebooks/energy_price_decade.ipynb
```

The original two-year pilot remains in results/ and its original notebook. The
extended study lives in results/decade and notebooks/energy_price_decade.ipynb.
`scripts/index_decade_data.py` rebuilds the checksum inventory after a rerun.
See the root README for methods and limitations. Model files are usable with the
recorded package versions; all fitting code and original inputs are provided.
