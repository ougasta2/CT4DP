# Danish electricity price association pilot

Reproducible DTU 02807 pilot: which realized wind/solar/load regimes accompany
negative or high day-ahead prices in DK1 and DK2, and do 2023 discoveries transfer
to 2024? This is exploratory association analysis, not causal analysis or a
deployable forecasting model. Technical deliverables are in English.

## Read the shared results

- [Executed notebook](notebooks/energy_price_pilot.ipynb): saved tables and figures;
  viewing it on GitHub does not require Python or API access.
- [Pilot findings](results/pilot_findings.md): concise conclusions and limitations.
- [Figures](results/figures/) and [execution record](results/validation.md).

The repository contains reusable code, tests, the executed notebook, summary
tables, fitted-model parameters and figures. Local virtual environments, raw API
caches, temporary smoke outputs and rebuildable row-level exports are excluded
by `.gitignore`; they remain on the original computer.

**A fresh clone has no data cache.** Follow the setup below and run the live full
pilot once before executing the notebook. The notebook intentionally uses
`--offline` and will report a cache miss until that download has completed.
The API can require several minutes of rate-limit waiting. To reproduce without
API access, obtain the original `data/cache/` directory separately and place it
at that same repository-relative path; then use `--offline`.

## Run (PowerShell, Python 3.10+)

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/run_pilot.py --smoke-test
.venv/Scripts/python scripts/run_pilot.py --train-year 2023 --test-year 2024
.venv/Scripts/python scripts/run_pilot.py --offline
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/execute_notebook.py
```

`scripts/build_notebook.py` is a maintainer tool that rebuilds the notebook
structure and clears saved outputs. It is not needed to read or rerun the
committed notebook.

Run commands from the repository root. No credentials required. The notebook
reruns the pipeline from cached official inputs and displays the resulting tables
and figures. Live first runs can take several minutes because the discontinued
price API has restrictive rate limits. Requests retry at most four times and
honor numeric `Retry-After` values up to ten minutes; longer waits fail visibly.
Connection/timeouts and server errors have bounded retries. Re-running reuses
completed cache entries. `--offline` makes no network requests and fails clearly
on a cache miss. `--output` and `--cache` select alternate directories.

## Inputs and attribution

**Energinet, Energi Data Service, CC BY 4.0.** Derived analyses and figures must
retain this attribution. Official references:

- [ProductionConsumptionSettlement metadata](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement)
- [Elspotprices metadata](https://api.energidataservice.dk/meta/dataset/Elspotprices)
- [API guide](https://www.energidataservice.dk/guides/api-guides)
- [Terms and conditions](https://www.energidataservice.dk/terms-and-conditions)
- [Support and service](https://www.energidataservice.dk/support-service)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The client verifies required columns against live metadata before downloading.
Every cached JSON envelope stores the exact URL, parameters, UTC retrieval time
and original payload (including schema, counts and records). Metadata is cached
too. Bounds are **Copenhagen local dates**, as specified by the API. `HourUTC`
values are parsed explicitly as UTC; all merges use `(HourUTC, PriceArea)`.
Years and calendar bins are derived in Europe/Copenhagen. Annual windows are
bounded and ordered, below the 50,000-record request limit; truncated responses
fail rather than silently omitting pages. Monthly coverage is audited against a
complete UTC hourly grid derived from local bounds, including DST and leap day.
Only the historical hourly price dataset is used (no post-2025 substitution).

The full source grids should contain 17,520 area-hours in 2023 and 17,568 in 2024.
`results/coverage.csv` checks these by source, month and area. Merge audits include
unmatched counts, per-column missingness, invalid exclusions and retention.
Raw cache is ignored by Git; retain it locally for reproducibility. Installed
versions are recorded in `requirements-lock.txt`; use it for the exact environment.

## Methods and scientific choices

- Wind is the sum of four official components; grid solar sums three grid
  components. All are MWh per hourly interval. Prices are EUR/MWh. Missing
  components never become zero. Nonfinite values, negative physical generation
  and nonpositive gross load are excluded and counted. Negative prices are kept.
- Gross consumption includes grid losses and is calculated from production,
  self-consumption and exchange. Metadata flags uncertain commercial
  self-consumption overlapping solar self-consumption. It is therefore a proxy
  for demand with accounting dependence on generation, not independent demand.
  Reported zeros are retained; source metadata warns some zeros can mean missing
  data, which cannot be resolved from the provided columns.
- Only 2023 fits area load medians, area demand tertiles, area 95th-percentile
  high-price thresholds, standardization and K-means. Ratios are not clipped.
  Features: wind/load, grid-solar/load and area-median load index; no price input.
  K=3..7 is compared by sampled silhouette, Davies–Bouldin, minimum cluster size
  and two-seed adjusted Rand stability. Highest silhouette among solutions with
  all clusters >=3% is selected. Centers and diagnostics remain inspectable.
- **Course topics:** sklearn K-means plus independently implemented Apriori.
  Apriori self-joins frequent itemsets, prunes candidates with missing frequent
  subsets, rejects mutually exclusive category items, and counts exact vertical
  intersections. Unit tests compare with hand counts and exhaustive enumeration.
  Antecedents contain at most three items and must have >=2% training support.
  Outcome events are scored **after predictor-only mining**, preserving rare
  outcome rules. Candidate scores retain support, joint support, confidence,
  lift and counts. The shortlist requires cluster+area, >=100 matching hours and
  >=20 event hours; select the highest training lift per area and event (max 4).
- Rules, preprocessing and thresholds are frozen in `frozen_training.json`
  before holdout scoring. Lift uses the same-year baseline of the rule's area.
  Calendar-only and calendar+cluster probabilities use 2023 cell rates, with
  training area-rate fallback for unseen cells; compare 2024 Brier scores.
  These are descriptive probability comparisons using realized energy.
- **Method beyond the course topics:** moving-block bootstrap of consecutive
  seven-day blocks, 400 deterministic replicates, both areas resampled together.
  DST days retain all observed hours. Missing dates remain in the sampling grid.
  Both rule rate and baseline are recomputed each replicate. Percentile 95%
  intervals require >=80% finite replicates; zero baselines yield undefined lift.
- Solar sensitivity adds estimated self-consumption. A full training-only
  refit is stored in `results/sensitivity/` (cluster IDs are not comparable), plus
  `sensitivity_comparison.csv` applies the primary frozen centroids and rules to
  perturbed holdout features. Missing sensitivity values are excluded and counted.
  This is a robustness check, never a better measurement silently substituted.

## Outputs and interpretation

`results/quality_profiles.csv` exposes missing/zero counts and distribution tails
for each year/area; extreme ratios and prices are retained rather than automatically
trimmed. `results/pilot_findings.md` is the concise empirical readout, not the final course
report. CSVs preserve denominators, diagnostics and candidate rules. Four primary
figures show a preselected first validation week, energy/negative-price regimes,
cluster profiles/sizes and fixed-rule lifts with exploratory uncertainty.
`notebooks/energy_price_pilot.ipynb` is the runnable companion. The smoke test
also exercises clustering, mining and plotting on a temporary four-day/three-day
split of the actual seven-day sample in `results/smoke_only/`; its synthetic
split labels do **not** represent 2023/2024 scientific evidence.

Serial dependence, overlapping selected rules, rare outcomes, many candidate
comparisons, revisions, missing-value ambiguity and omitted exchange/weather/fuel
variables limit conclusions. Bootstrap intervals do not correct discovery bias.
An association in generation settled after market clearing is not information
available to a day-ahead predictor. A lower descriptive Brier score is not a
claim of causal or operational forecasting performance. Holdout results must not
be used to retune selection. Record failures in `last_failure.json`; a failed run
does not certify existing outputs as current.

Plan against **December 4, 2026**; verify on DTU Learn because the supplied
assignment description also mentions December 6. No final report is written here.
