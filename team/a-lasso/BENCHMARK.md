# Shared DK1 benchmark

Install this repository's dependencies in a Python 3.10+ environment and run
the commands in its README. The shared snapshot is included in `data/` and
requires no API keys or downloads. See the [data sources and licence](DATA_LICENSE.md).
Keep experiment outputs in the Git-ignored `results/` directory.

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
