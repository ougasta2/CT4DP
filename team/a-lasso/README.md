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
