# D. Multi-layer perceptron

Build a feed-forward neural network that predicts all 24 hourly DK1 day-ahead
prices together. Test whether its shared hidden layers improve on the project
lead's frozen Ridge baseline on the same delivery hours. This model is the
supervised learning extension. Its aligned predictions can also be used in the
team's clustering and Apriori analyses.

Run the [shared benchmark](../BENCHMARK.md) with `--model mlp`. Outputs are in `results/team/mlp/`.

## What to investigate

1. Record the unchanged starter's results. It has two hidden layers (80 and
   40 units), L2 penalty `alpha=1.0`, a 400-iteration budget and a fixed random
   seed. Inputs and each hourly target are standardized using the current
   training window only. Predictions are converted back to EUR/MWh.
2. Compare one change at a time, starting with a smaller network such as
   `(40,)` or a different L2 penalty. Use chronological validation when
   choosing settings. The MLP's automatic early stopping uses a randomly
   split validation subset, so it is disabled in the starter. Record any
   convergence warning, its impact on runtime and the results of any
   larger iteration budget.
3. Compare with Ridge on matching timestamps. Report MAE, RMSE, the ±5% hit
   count and eligible count, negative-price MAE and its sample size, and
   fit time. Check whether any change is concentrated at particular hours.
   The two benchmark years are historical development data.
4. Send the project lead your model code, settings and a short account of
   the hypothesis, results, limitations and your individual contribution.
   Include `results/team/mlp/metrics.csv`, `results/team/mlp/timing.csv`, and
   `results/team/mlp/run.json`, and transfer `results/team/mlp/predictions.csv.gz` through an
   approved private channel when its use is authorized.

Edit `model.py` while preserving `build_model()`: it must return an unfitted
scikit-learn estimator that accepts one feature row per day and predicts 24
finite values per row. Leave the shared data, evaluation periods and output
schema unchanged for the main comparison.
