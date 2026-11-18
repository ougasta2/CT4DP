# DK1 day-ahead electricity price forecasting

A team research project on predicting the hourly mean wholesale electricity price in western Denmark (DK1). Four contributors can independently improve model families using the same data and walk-forward evaluation; the project lead integrates the forecasts and coordinates the report.

**Current state:** the team benchmark is a new starting framework. Its four model implementations have not yet been established as superior or fully compared. Earlier experiments and results remain available. Historical inputs have not all been audited for release time and revisions, and every historical year used so far has been inspected. The work is exploratory, not a verified operational forecasting system.

## Start here

1. Read [the team workflow and experimental protocol](docs/TEAM_WORKFLOW.md).
2. Check the [data catalog and setup](docs/DATA_CATALOG.md), including files that cannot be redistributed in Git.
3. Find your [assigned model folder and branch](team/README.md).
4. Run the shared benchmark and report your change, metrics and runtime.

## Team benchmark

The lead's fixed Ridge baseline has completed the January 2023–December 2024
comparison. See [Ridge benchmark results](models/ridge/README.md).

The four member workspaces are consolidated under [team/](team/README.md). See the team workflow for branch boundaries and pull requests.

From the repository root, with Python 3.10+:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/run_team_benchmark.py --model ridge --start 2023-01-01 --end 2023-02-01
```

Replace `ridge` with `lasso`, `random_forest`, or `mlp`. The default period is January 2023 through December 2024; `--end` is exclusive and dates must be month starts. The example runs one month as a setup check. The full benchmark takes much longer, particularly for the neural network and forest. Each run overwrites that model's files under `results/team/<model>/`; save your comparison elsewhere before rerunning. This directory is Git-ignored because row-level forecasts may embed third-party data. A run needs the inputs described in the catalog; installing dependencies alone is not sufficient.

After all four models have been run for exactly the same period, create an equal-weight baseline:

```powershell
.venv/Scripts/python scripts/ensemble_team.py results/team/ridge/predictions.csv.gz results/team/lasso/predictions.csv.gz results/team/random_forest/predictions.csv.gz results/team/mlp/predictions.csv.gz
```

The ensemble script checks that timestamps, actual prices and local day/hour agree exactly. It writes `results/team/ensemble/predictions.csv.gz`. It also writes annual metrics using the common definitions. It does not choose weights. Any learned weights must be selected on earlier out-of-fold forecasts only.

```powershell
.venv/Scripts/python -m pytest -q
```

## Repository map

| Location | Purpose |
| --- | --- |
| [team/](team/README.md) | A–D model workspaces, assigned branches, and pull request workflow |
| [models/](models/README.md) | Original model implementations used by the root benchmark runner |
| [scripts/run_team_benchmark.py](scripts/run_team_benchmark.py) | Shared data preparation, monthly evaluation, metrics and timing |
| [scripts/ensemble_team.py](scripts/ensemble_team.py) | Alignment-checked equal-weight ensemble |
| [data/](docs/DATA_CATALOG.md) | Source-specific inputs, caches and data provenance |
| [results/](results/README.md) | Prior experiments and ignored team-run outputs |
| [docs/](docs/README.md) | Team protocol, data catalog and historical research documentation |
| [scripts/](scripts/README.md) | Team entry points and prior experimental pipelines |
| [src/](src/energy_pilot/) and [notebooks/](notebooks/) | Earlier reusable analysis and executed pilot notebooks |
| [tests/](tests/) | Regression and pipeline checks |

Source files remain in their existing folders so the historical scripts and notebooks continue to work. See [historical research status](results/dk1_demand_price/RESEARCH_STATUS.md), [rolling benchmark results](results/dk1_rolling_benchmark/findings.md), and [the original pilot](results/pilot_findings.md). These describe different experiments and must not be compared as if they shared a protocol.

## Attribution

The study uses Energinet Energi Data Service data under CC BY 4.0, plus third-party price and weather sources. Preserve each source's attribution and terms; see the [data catalog](docs/DATA_CATALOG.md) and source-specific READMEs. Some source files and derivatives remain local pending redistribution review.
