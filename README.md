# B · Random forest and market clustering

You own two experiments that can run independently. The forest forecasts the
next day's 24 hourly DK1 prices. The Week 6 experiment groups market days using
features available at the assumed forecast cutoff and supplies regime labels for
the team's error analysis.

Create a Python 3.10+ environment, install `requirements.txt`, and place the
team's approved benchmark files in `data/` as described in the shared protocol.
From this repository's root, run the one-month integration check and then the
2023–2024 walk-forward benchmark:

```bash
python -m pip install -r requirements.txt
python benchmark.py --start 2023-01-01 --end 2023-02-01
python benchmark.py
```

The full run refits the forest 24 times. Archive its local
`results/metrics.csv`, `timing.csv`, `run.json`, and `predictions.csv.gz` before
running another experiment into the same folder.

1. Run the fixed forest in `model.py`. Keep the daily input and 24-output
   interface. Try one change, such as moving `min_samples_leaf` from 5 to 10 to
   test whether more averaging reduces large errors. Make choices on earlier
   data and compare on identical hours. Record annual MAE, eligible ±5% hit
   fractions, negative-price error counts and fit/predict time.
2. Run the fixed four-cluster starter with `python cluster_regimes.py`. It uses
   DK1's previous-day price shape and archived DK1 weather forecasts. Scaling
   and K-means are fit on days before 2023. Today's price, targets and forecast
   errors do not enter the fit.
3. Compare candidate choices of `k` on pre-2023 data using the Davies–Bouldin
   index (lower is better), cluster sizes and interpretable centers. Use
   `--output-dir` for separate local experiment outputs. Explain what the
   clusters represent, and check whether their sizes and error patterns change
   from 2023 to 2024.
4. Share the selected `results/regimes.csv` with the lead and C through the
   team's approved data route. It contains a unique Copenhagen local date
   (`day`) and an integer `cluster` for each day in 2023–2024. Keep
   `regime_centers.csv` and `regime_diagnostics.csv` for the write-up. Join labels
   to benchmark predictions by local `day` to report error by regime. Use only
   historical predictions to estimate any correction for later evaluation.

Deliver the runnable code, model settings, dates, scores with denominators, and
a paragraph interpreting the forest and clustering results. These are
historical development years and archived inputs have incomplete point-in-time
verification. Keep benchmark inputs and row-level outputs local until the
source terms and approved sharing route have been checked.
