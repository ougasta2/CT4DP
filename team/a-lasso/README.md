# A · Lasso

Lasso fits a penalized linear forecast for each of the 24 local delivery hours.
Its L1 penalty can set coefficients to zero. Use it to test whether sparse
selection of lagged prices and weather features improves the DK1 forecast.

Run the [shared benchmark](../BENCHMARK.md) with `--model lasso`. Outputs are in `results/team/lasso/`.

1. Compare the unchanged starter (`alpha=0.03`) with the project lead's fixed
   Ridge results over the same delivery hours. Report MAE, the ±5% hit rate
   and its eligible count, negative-price errors, and runtime.
2. Test a small, prespecified grid for `alpha`, using only earlier rolling
   validation months to make the choice. Report changes in coefficient
   sparsity and validation error. If optimization emits a convergence warning,
   check the number of iterations and record whether increasing the limit
   changes the result.
3. Evaluate the chosen setting on later months using the same input snapshot
   and runner. Keep learned scaling and any feature selection inside the
   estimator in `model.py`. The benchmark calls `build_model()` for each fit
   and expects one row with 24 predictions for each delivery day.

Send the lead your model revision, settings tried, benchmark outputs through
the agreed data-sharing channel, and a short explanation of the result.
The lead will reuse aligned predictions for the shared clustering, Apriori,
and ensemble analyses. Keep row-level predictions out of Git.

## Delivered by A

Decision: **alpha = 0.3** with `max_iter` = 10000 in `model.py` (starter: alpha = 0.03,
`max_iter` = 3000). Scaling stays inside the pipeline. All findings are exploratory:
2023–2024 are development years.

| File | Purpose |
| --- | --- |
| `lasso.ipynb` | Evidence for every choice, with outputs saved. It follows the course exercise *Ridge and Lasso Regression*: chronological split (train 2019–2022, validation 2023, test 2024), unregularized baseline (Part 2), feature-set tuning (Part 3), Lasso alpha grid and feature selection (Part 4), Ridge alpha grid (Part 5), comparison (Part 6), and the shared benchmark with the chosen alpha (Part 7). |
| `model.py` | The chosen Lasso, written by Part 7 of the notebook. |
| `temporaryresults/` | The row-level hand-off files (`predictions.csv.gz`, `metrics.csv`, `timing.csv`); temporary, see below. |

Run order, from `team/a-lasso/` with the project environment: open `lasso.ipynb` and run
all cells (about 20 minutes; the rolling alpha grid in Part 4 takes most of it). Part 7 calls the shared benchmark for `ridge` and `lasso`;
outputs are in the Git-ignored `results/team/<model>/`. The unchanged starter was run with
`python scripts/run_team_benchmark.py --model lasso --data --output results/team/lasso-starter`.

**Alpha choice.** The grid {0.03, 0.1, 0.3, 1, 3, 10, 30, 100} was fixed in advance and
evaluated on rolling validation months only: for each month of 2023 the scaler and the
Lasso were refitted on all earlier days and predicted that month (Part 4 of the notebook).
The selection criterion is the validation RMSE, as in the course exercise; 2024 was not
used for any choice.

| alpha | Validation RMSE | Validation MAE | Zero coefficients (of 14,496, mean of 12 fits) |
| ---: | ---: | ---: | ---: |
| 0.03 (starter) | 29.43 | 20.65 | 8,754 (60%) |
| 0.1 | 27.33 | 18.98 | 11,099 (77%) |
| **0.3** | **26.14** | 17.91 | 12,655 (87%) |
| 1 | 26.34 | 17.81 | 13,549 (93%) |
| 3 | 28.05 | 18.97 | 13,992 (97%) |
| 10 | 31.19 | 21.38 | 14,229 (98%) |
| 30 | 35.05 | 24.82 | 14,382 (99%) |
| 100 | 46.02 | 34.62 | 14,479 (99.9%) |

alpha = 0.3 has the lowest validation RMSE. alpha = 1 is practically tied and has a slightly
lower validation MAE (17.81 vs 17.91), so the choice between the two depends on the
criterion; RMSE was fixed beforehand. Parts 2–6 of the notebook also compare unregularized
linear regression, Ridge and Lasso on a fixed split (fit on 2019–2022): the unregularized
model overfits (training RMSE 18.1, validation 37.8).

**Convergence.** The starter emits ConvergenceWarnings because of its iteration limit: on the
2019–2022 window at alpha = 0.03, coordinate descent needs 5,762 iterations, more than the
starter's 3,000 (21 of the 24 hour models stop early). Raising the limit to 10,000 removes
all warnings but does not change the result: validation RMSE 31.028 vs 31.030, mean
prediction difference 0.04 EUR/MWh (max 0.81). The chosen model uses `max_iter` = 10000, and
no alpha in the grid emits a warning with it. Small alphas are slow because the 604
standardized inputs are strongly collinear (24 hourly values per variable, correlated
neighbouring zones).

**Shared benchmark** (monthly expanding-window refits, hourly scores, same hours for all
models):

| Model | Year | MAE | RMSE | ±5% hits / eligible | Negative-price MAE (hours) | Fit time, 24 refits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge, alpha = 100 (frozen) | 2023 | 18.78 | 26.65 | 1,575 / 7,738 (20.4%) | 31.56 (281) | 3 s |
| Lasso starter, alpha = 0.03 | 2023 | 20.65 | 29.43 | 1,500 / 7,738 (19.4%) | 33.88 (281) | 744 s |
| **Lasso, alpha = 0.3** | 2023 | **17.91** | **26.14** | 1,691 / 7,738 (21.9%) | 30.94 (281) | 139 s |
| Ridge, alpha = 100 (frozen) | 2024 | 18.34 | 29.52 | 1,356 / 7,620 (17.8%) | 19.43 (375) | 3 s |
| Lasso starter, alpha = 0.03 | 2024 | 19.14 | 30.31 | 1,270 / 7,620 (16.7%) | 19.82 (375) | (same run) |
| **Lasso, alpha = 0.3** | 2024 | **17.84** | **29.35** | 1,429 / 7,620 (18.8%) | 19.79 (375) | (same run) |

Fit times are for both years together, on a Windows laptop with an AMD Ryzen AI 9 HX 370 (the runner limits BLAS to 2 threads).

**What I learned.** Sparsity helps: the chosen Lasso is better than the Ridge baseline on
MAE, RMSE and the ±5% hit rate in both years, while using about 77 of 604 inputs per hour
model; the starter (almost no penalty) is worse than Ridge and much slower. Negative-price
hours remain the weak point: errors there stay large for both models, and Lasso is slightly
worse than Ridge in 2024. Lasso also costs about 45 times Ridge's fit time. Lasso keeps mostly lagged prices of SE4 and
NL, then wind speed and cloud cover; temperature is almost never used. German prices are
kept but are not dominant: DE-LU `lag1` and `lag7` have about 3 non-zero coefficients per
hour model each (of 24), and `DE-LU_lag1_h23` is used by 15 of the 24 hour models. Because
the inputs are collinear, which individual hour of a correlated variable is kept is not
stable; conclusions should be read at the level of variables, not single hours.

Row-level files, included **temporarily** for the lead and C to collect, in
`temporaryresults/`:
- `predictions.csv.gz`: hourly forecasts and actual prices from the chosen Lasso (alpha = 0.3), 2023–2024, for the clustering, Apriori and ensemble analyses.
- `metrics.csv` and `timing.csv`: annual metrics and per-month fit/predict times of the same run.

The team convention is to keep row-level outputs in the Git-ignored `results/team/`
directory, so the lead should remove `temporaryresults/` once the files are collected.
The same files are regenerated by Part 7 of the notebook. Price and weather data are
adapted from the sources and licences listed in `../DATA_LICENSE.md`.
