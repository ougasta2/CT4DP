# Frozen Ridge baseline

`model.py` defines the project lead's baseline: training-only standardization
followed by multi-output Ridge with the fixed `alpha=100`. The model predicts
24 local clock-hour prices and is refit monthly on expanding history.

Run from the repository root:

```powershell
.venv/Scripts/python scripts/run_team_benchmark.py --model ridge --start 2023-01-01 --end 2025-01-01
```

Full benchmark, with the intended 10:00 Copenhagen issue time on D-1:

| Year | Evaluated hours | MAE (EUR/MWh) | ±5% hits / eligible hours |
| --- | ---: | ---: | ---: |
| 2023 | 8,760 | 18.78 | 1,575 / 7,738 (20.35%) |
| 2024 | 8,784 | 18.34 | 1,356 / 7,620 (17.80%) |

Eligibility means absolute actual price of at least EUR 20/MWh. Both years
are historical development. Input publication and revision times have only
been partially audited. A rerun overwrites local files in `results/team/ridge/`.
See the [team protocol](../../docs/TEAM_WORKFLOW.md) and [data catalog](../../docs/DATA_CATALOG.md).
