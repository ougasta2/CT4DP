# B · Random forest and market clustering

You own two experiments that can run independently. The forest forecasts the
next day's 24 hourly DK1 prices. The Week 6 experiment groups market days using
features available at the assumed forecast cutoff and supplies regime labels for
the team's error analysis.

Run the [shared benchmark](../BENCHMARK.md) with `--model random_forest`. Outputs are in `results/team/random_forest/`.

1. Run the fixed forest in `model.py`. Keep the daily input and 24-output
   interface. Try one change, such as moving `min_samples_leaf` from 5 to 10 to
   test whether more averaging reduces large errors. Make choices on earlier
   data and compare on identical hours. Record annual MAE, eligible ±5% hit
   fractions, negative-price error counts and fit/predict time.
2. Run the fixed four-cluster starter with `python team/b-random-forest/cluster_regimes.py`. It uses
   DK1's previous-day price shape and archived DK1 weather forecasts. Scaling
   and K-means are fit on days before 2023. Today's price, targets and forecast
   errors do not enter the fit.
3. Compare candidate choices of `k` on pre-2023 data using the Davies–Bouldin
   index (lower is better), cluster sizes and interpretable centers. Use
   `--output-dir` for separate local experiment outputs. Explain what the
   clusters represent, and check whether their sizes and error patterns change
   from 2023 to 2024.
4. Share the selected `results/team/random_forest/regimes.csv` with the lead and C through the
   team's approved data route. It contains a unique Copenhagen local date
   (`day`) and an integer `cluster` for each day in 2023–2024. Keep
   `regime_centers.csv` and `regime_diagnostics.csv` for the write-up. Join labels
   to benchmark predictions by local `day` to report error by regime. Use only
   historical predictions to estimate any correction for later evaluation.

Deliver the runnable code, model settings, dates, scores with denominators, and
a paragraph interpreting the forest and clustering results. These are
historical development years and archived inputs have incomplete point-in-time
verification. Keep row-level outputs in the Git-ignored results directory.
