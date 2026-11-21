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

Run order, from `team/a-lasso/` with the project environment: open `lasso.ipynb` and run
all cells (about 5 minutes). Part 7 calls the shared benchmark for `ridge` and `lasso`;
outputs are in the Git-ignored `results/team/<model>/`. The unchanged starter was run with
`python scripts/run_team_benchmark.py --model lasso --data --output results/team/lasso-starter`.

**Alpha choice.** The grid {0.03, 0.1, 0.3, 1, 3, 10, 30, 100} was fixed in advance. Each
model was fitted on 2019–2022 and scored on 2023; alpha = 0.3 had the lowest validation
RMSE (26.62, alpha = 1: 26.65). 2024 was not used for any choice. With the same split,
the unregularized linear regression overfits (training RMSE 18.1, validation 37.8),
Ridge chooses alpha = 1000 (validation 27.19) and Lasso keeps 1,847 of 14,496 coefficients
(87% exactly zero).

**Convergence.** The starter's ConvergenceWarnings come from the iteration limit: on the
2019–2022 window, coordinate descent needs about 5,800 iterations at alpha = 0.03, more
than the starter's 3,000. With `max_iter` = 10000 every alpha in the grid converges.
Small alphas are slow because the 604 standardized inputs are strongly collinear
(24 hourly values per variable, correlated neighbouring zones).

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

Row-level predictions are not in Git: `results/team/lasso/predictions.csv.gz`,
`metrics.csv` and `timing.csv` are sent to the lead through the team channel. Price and
weather data are adapted from the sources and licences listed in `../DATA_LICENSE.md`.
