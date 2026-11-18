# Random forest

`model.py` is this member's editable starting point. `build_model()` must return an unfitted estimator that accepts the shared daily features and predicts 24 hourly values. Put learned preprocessing inside the estimator. Study nonlinear interactions, tree depth, leaf size and computational cost.

Run from the repository root:

```powershell
.venv/Scripts/python scripts/run_team_benchmark.py --model random_forest --start 2023-01-01 --end 2023-02-01
```

Save the baseline before modifying the model; a rerun overwrites `results/team/random_forest/`. Use the same full comparison period as the rest of the team and record the hypothesis, code version, metrics, timing, results and limitations. See the [team protocol](../../docs/TEAM_WORKFLOW.md) and [data catalog](../../docs/DATA_CATALOG.md).
