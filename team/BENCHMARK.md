# Shared DK1 benchmark

From the repository root, with a Python 3.10+ virtual environment active:

```sh
python -m pip install -r requirements.txt
python scripts/run_team_benchmark.py --model lasso --data --start 2023-01-01 --end 2023-02-01
```

Choose `lasso`, `random_forest`, `gradient_boosting`, `mlp` or `ridge`.
Remove the dates for a full run. The snapshot is in `team/data/` and needs
no API keys or downloads. See the [data sources and licence](DATA_LICENSE.md).
By default outputs go to the ignored `results/team/<model>/` directory.

The model API is `build_model()`: return a fresh estimator with
`fit(features, targets)` and `predict(features)`. Each row is one Copenhagen
delivery day. Return 24 finite prices in EUR/MWh, ordered by local clock hour.
The benchmark refits monthly on expanding history, with no evaluation days in
the fit. It scores the original hourly observations, including the two
observations at the repeated fall clock hour. Keep any fitted transforms in
the model pipeline.

The full run covers January 2023 through December 2024. Both years are
historical development. The issue time is intended to be 10:00 Copenhagen
time on D-1; release and revision times are only partially audited. Use
predictions from 2023 to choose error rules or ensemble weights. Freeze choices
before evaluating them on 2024. Label the results as exploratory.

Report all-hour MAE and RMSE and the ±5% hit rate for hours with an absolute
actual price of at least EUR 20/MWh. Report eligible counts and negative-price
MAE. Include fit and prediction time, hardware and the dataset manifest.
Compare against the [frozen Ridge result](RIDGE_BASELINE.md) using the same
period and hourly targets. Each full run writes aligned `predictions.csv.gz`,
`metrics.csv`, `timing.csv` and `run.json`. Use `--output results/experiment-name`
to preserve a baseline while running an experiment.

Hand back your code revision, English write-up, annual metrics and timing,
and the aligned prediction file through the agreed team channel. Use the
row-level output for the common clustering, Apriori and ensemble analyses.
