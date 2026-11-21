# C · Gradient boosting and Apriori: results (first pass)

Exploratory historical analysis. Both 2023 and 2024 are development years
already inspected in earlier work; nothing here is an independent test.
Settings were chosen on 2023 only, and 2024 is reported as a check.

## Pipeline

```text
# 1. Unchanged starter (preserved baseline)
python scripts/run_team_benchmark.py --model gradient_boosting --data --output results/team/gradient_boosting_baseline
# 2. One-change variants (same evaluator, inputs, dates and metrics)
python team/c-gradient-boosting/run_experiments.py smaller_leaves   # also: strong_l2, absolute_loss
# 3. Reference model for cross-model error patterns
python scripts/run_team_benchmark.py --model ridge --data
# 4. Shipped Apriori analysis (strongest rule per group)
python team/c-gradient-boosting/analyze_apriori.py --predictions results/team/gradient_boosting_baseline/predictions.csv.gz
# 5. Comparison table + every rule with 2023 and 2024 scores, for all runs
python team/c-gradient-boosting/summarize.py
# 6. With B's cluster labels
python team/c-gradient-boosting/analyze_apriori.py --predictions results/team/gradient_boosting_baseline/predictions.csv.gz --regimes team/b-random-forest/temporaryresults/regimes.csv --output results/team/gradient_boosting_baseline/apriori_with_clusters.csv
python team/c-gradient-boosting/summarize.py --regimes team/b-random-forest/temporaryresults/regimes.csv
```

Environment: Python 3.13, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.5;
2 vCPU Intel Xeon 2.30 GHz (Linux), thread pools limited to 2 by the runner.
Repository revision `bc6bd20` (main after B's merge) plus this folder's scripts, bundled snapshot `team/data/`.

## Forecast experiment

Starter: one `HistGradientBoostingRegressor` per local clock hour
(60 iterations, learning rate 0.08, 15 leaves, 30 samples per leaf, L2 = 1).
Each variant changes exactly one setting.

| Run | Change | Year | MAE | RMSE | ±5% hits / eligible | Neg. MAE (n) | Fit s / year |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge (lead) | — | 2023 | 18.78 | 26.65 | 1,575 / 7,738 (20.4%) | 31.56 (281) | 0.9 |
| | | 2024 | 18.34 | 29.52 | 1,356 / 7,620 (17.8%) | 19.43 (375) | 1.1 |
| **GB starter** | — | 2023 | **18.52** | 27.02 | 1,709 / 7,738 (22.1%) | 36.06 (281) | 183 |
| | | 2024 | **16.57** | 27.52 | 1,462 / 7,620 (19.2%) | 20.54 (375) | 189 |
| GB smaller_leaves | `max_leaf_nodes` 15→7 | 2023 | 19.02 | 27.50 | 1,659 (21.4%) | 39.28 | 130 |
| | | 2024 | 17.73 | 28.88 | 1,341 (17.6%) | 25.18 | 132 |
| GB strong_l2 | `l2_regularization` 1→10 | 2023 | 18.74 | 27.37 | 1,703 (22.0%) | 37.49 | 175 |
| | | 2024 | 16.59 | 27.62 | 1,405 (18.4%) | 20.98 | 187 |
| GB absolute_loss | `loss` squared→absolute | 2023 | 19.24 | 27.91 | 1,652 (21.3%) | 39.45 | 187 |
| | | 2024 | 17.32 | 29.10 | 1,532 (20.1%) | 26.39 | 195 |

Hours: 8,760 (2023) and 8,784 (2024) for every run. Fit time is the summed
monthly fit wall time; full-run wall time was about 4.5–6.5 minutes per GB run.

**Decision (on 2023): keep the starter settings.** None of the three changes
improved 2023 MAE or the 2023 hit rate. Gradient boosting beats Ridge on MAE
in both years (−0.26 and −1.77 EUR/MWh) and on the hit rate, but is worse on
negative-price hours in both years and costs ~180× more fit time.
Smaller trees cut fit time by ~29% at a cost of 0.5 (2023) to 1.2 (2024) EUR/MWh MAE.
Absolute-error loss gave the best 2024 hit rate, but it was worse on 2023 and
it cannot be chosen from 2024, so this is reported only as an observation.

## Apriori experiment

Large error = |error| above the 2023 90th percentile of that model's errors
(~41 EUR/MWh). Baskets: season × time of day. Minimum support 5%, at least 20
large errors in 2023.

Shipped strongest-rule output for the GB starter:

| Rule | 2023 support / confidence / lift | 2024 support / confidence / lift |
| --- | --- | --- |
| spring & afternoon | 6.3% / 21.6% / 2.16 | 6.3% / 4.5% / 0.72 |

The strongest 2023 rule **does not hold in 2024**. Looking at every rule
(`summarize.py`) shows a more useful picture:

| Rule | Lift 2023 → 2024, GB starter | Ridge | Holds in all 5 runs? |
| --- | --- | --- | --- |
| autumn & evening | 1.98 → 2.07 | 2.18 → 1.88 | yes (2024 lift 1.88–2.07) |
| autumn & morning | 1.58 → 1.37 | 1.70 → 1.57 | yes |
| afternoon (any season) | 1.46 → 1.33 | 1.31 → 1.17 | yes |
| spring & afternoon | 2.16 → 0.72 | 1.99 → 0.63 | no in 4 of 5 (abs. loss: 1.10) |

Interpretation: autumn evenings and mornings are a **stable failure regime for
both model families**, possibly linked to the evening demand ramp after solar output
drops (not tested here). Spring afternoons were a 2023-specific
problem (the solar/negative-price hours), which every model handled better in
2024. Selecting a rule by highest 2023 lift alone picks the unstable one, so
lift should be read together with year-to-year stability. Because the error
threshold is fixed from 2023 and the GB model improved, the 2024 base rate is
6.3% rather than 10%; lift accounts for this.

### Adding B's cluster labels

Labels: `team/b-random-forest/temporaryresults/regimes.csv` (k = 4, fitted on
days before 2023 from the previous day's price shape and DK1 weather
forecasts). From B's centres: cluster 0 = windy and cloudy, cluster 1 = calm
and clear, cluster 2 = mild and calm, cluster 3 = the day after a very
expensive day (mean previous-day price ~293 EUR/MWh vs ~60–66 for the others).

Shipped strongest-rule output for the GB starter:

| Group | Rule | 2023 support / confidence / lift | 2024 support / confidence / lift |
| --- | --- | --- | --- |
| calendar | spring & afternoon | 6.3% / 21.6% / 2.16 | 6.3% / 4.5% / 0.72 |
| cluster | cluster 1 & afternoon | 5.5% / 17.5% / 1.75 | 4.1% / 6.1% / 0.97 |

Neither selected rule holds in 2024. Among all 44 rules for the GB starter,
the cluster rules have lower 2023 lift than the best calendar rules, and the
only one that clearly holds is cluster 2 & autumn (lift 1.46 → 1.34). For
Ridge it falls to 0.96 in 2024, while cluster 0 & evening holds
(1.59 → 1.34). Cluster items therefore add little stable information beyond
season and time of day.

**Cluster 3 is invisible to Apriori.** It covers 11 days in 2023 and 18 in
2024 (under 5% of hours), so no itemset containing it reaches the minimum
support. A direct check outside the rule search shows it is the one cluster
with a consistently high large-error rate:

| Cluster | Days 2023 / 2024 | GB large-error rate 2023 / 2024 | GB MAE 2023 / 2024 | Ridge large-error rate 2023 / 2024 |
| --- | --- | --- | --- | --- |
| 0 | 129 / 131 | 10.7% / 4.5% | 19.3 / 14.8 | 12.1% / 8.0% |
| 1 | 80 / 60 | 9.2% / 5.2% | 17.5 / 14.6 | 5.0% / 6.0% |
| 2 | 145 / 157 | 9.3% / 7.0% | 18.0 / 17.0 | 10.3% / 8.9% |
| 3 | 11 / 18 | 17.4% / 16.7% | 23.7 / 32.5 | 18.6% / 25.9% |
| all | 365 / 366 | 10.0% / 6.3% | 18.5 / 16.6 | 10.0% / 9.0% |

For GB that is lift ~1.7 in 2023 and ~2.6 in 2024; for Ridge ~1.9 and ~2.9.
Both models miss most after price spikes, but this is based on very few days
and was found by inspection, not by the pre-set rule search. The support
threshold is a real limitation: rare regimes are often the costly ones.
Lowering it would be a new, separate experiment.

Caveats: many rules were searched, neighbouring hours are dependent, and
calendar items are coarse (each two-item rule covers ~6% of hours). No error
correction was attempted.

## Next

- Deliver `results/team/gradient_boosting_baseline/predictions.csv.gz` (the chosen run) to the lead; row-level files stay out of Git.
