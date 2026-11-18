# Results index

`team/` is the local, Git-ignored output of the new common runner. Each model receives `predictions.csv.gz`, `metrics.csv`, `timing.csv`, and `run.json`; running it again overwrites those files. The ensemble script writes aligned predictions and annual metrics under `team/ensemble/`.

Earlier studies retain their original paths to preserve reproducibility. Useful starting points are [rolling benchmark findings](dk1_rolling_benchmark/findings.md), [demand and price research status](dk1_demand_price/RESEARCH_STATUS.md), [weather ablation](dk1_weather_ablation/findings.md), and [ten-year findings](decade/pilot_findings.md). They use different samples, inputs and horizons; compare only after matching their protocols. Model files and row-level predictions involving restricted sources may be absent from a clone; consult [.gitignore](../.gitignore) and the [data catalog](../docs/DATA_CATALOG.md).
