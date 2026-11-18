# C · Gradient boosting and Apriori

Own the gradient-boosting forecast and the Week 5 frequent-itemset analysis
for next-day hourly DK1 prices. The forecast produces 24 local clock-hour
predictions through the shared expanding-window benchmark. The Apriori analysis
starts with your forecasts or any aligned full-period forecast, such as the
project lead's shared Ridge baseline.

Run the [shared benchmark](../BENCHMARK.md) with `--model gradient_boosting`. Outputs are in `results/team/gradient_boosting/`.

## Forecast experiment

The starter trains one histogram gradient-boosting regressor per local
clock hour. It uses the common features and assumes the team's 10:00
Copenhagen cutoff on the day before delivery. Begin by comparing the shipped
settings against one change, such as a smaller leaf limit or stronger leaf
regularization. An eligible-hour ±5% hit-rate gain may trade off against
all-hour MAE or errors in negative-price hours. Compare on identical hours,
report both scores and denominators, and include training time. Choose settings
using chronological development results preceding any evaluation used to
judge them. Keep finite prediction arrays of shape `(days, 24)` and leave
the evaluator's inputs, dates and metric definitions fixed.

## Apriori experiment

```text
python team/c-gradient-boosting/analyze_apriori.py
python team/c-gradient-boosting/analyze_apriori.py --regimes results/team/random_forest/regimes.csv --output results/team/gradient_boosting/apriori_with_clusters.csv
```

The first command can run before B delivers cluster labels. Use
`--predictions PATH` to analyze a different aligned forecast. The second
command adds B's CSV with one `day,cluster` row per Copenhagen delivery date.
Each prediction hour is a basket of predictor items: season, time of day and,
if supplied, the preassigned cluster. Apriori mines combinations covering at
least 5% of 2023 hours. Label an error large if its absolute value exceeds the
2023 90th percentile. Select the strongest rule by 2023 lift in each available
group (calendar-only and cluster-containing), requiring at least 20 large
errors. Apply the same rule and threshold to 2024. The output includes matched
hours, large-error counts, support (share of hours matching the rule),
confidence (large-error fraction among matches), overall event rate and lift
(confidence divided by that period's overall event rate). Check whether
higher lift comes at the expense of coverage or disappears in 2024.

Report this as exploratory historical error analysis. Many rules were
searched, nearby hours are dependent, and both years have been used in prior
work. A rule can help identify a failure regime even when a resulting
correction does not lower price MAE. Keep any later correction experiment
separate from rule selection and fit it on earlier prediction errors only.

## Hand back to the project lead

- Your estimator, experiment settings and a brief explanation of the changes.
- The baseline and chosen run's metrics, timing, metadata and aligned
  `results/team/gradient_boosting/predictions.csv.gz`, shared through the team's approved channel.
- The Apriori output and a short interpretation of both rule groups, including
  inconclusive or worse results. Keep row-level predictions out of version control.
