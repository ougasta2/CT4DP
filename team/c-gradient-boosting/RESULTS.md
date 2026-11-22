# C · Gradient boosting and Apriori: results

2023 is the validation year: the gradient-boosting settings, the Apriori
error threshold and the rules are all chosen on 2023 results. These choices are
then held fixed and scored on 2024, which is an exploratory evaluation (both
years were inspected in earlier project work, so it is not an untouched test).

## Reproducing

All numbers below come from the pinned environment:

```text
python -m pip install -r requirements-lock.txt     # scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3
# 1. Unchanged starter (preserved baseline)
python scripts/run_team_benchmark.py --model gradient_boosting --data --output results/team/gradient_boosting_baseline
# 2. One-change variants (same evaluator, inputs, dates and metrics)
python team/c-gradient-boosting/run_experiments.py smaller_leaves   # also: strong_l2, absolute_loss
# 3. Reference model for cross-model error patterns
python scripts/run_team_benchmark.py --model ridge --data
# 4. Shipped Apriori analysis (strongest rule per group), with B's cluster labels
python team/c-gradient-boosting/analyze_apriori.py --predictions results/team/gradient_boosting_baseline/predictions.csv.gz --regimes team/b-random-forest/temporaryresults/regimes.csv --output results/team/gradient_boosting_baseline/apriori_with_clusters.csv
# 5. Comparison table + every rule with 2023 and 2024 scores
python team/c-gradient-boosting/summarize.py --regimes team/b-random-forest/temporaryresults/regimes.csv
```

`summarize.py` summarizes the starter, the three variants and Ridge by default;
`--runs` takes other run names or folders. Before writing anything it checks
that every selected run covers all 17,544 hours of 2023–2024, and if one does
not, it names the run and prints the command to rerun it. A variant run over a
shorter period (for example `--end 2023-02-01`) is written to its own folder,
`gradient_boosting_<variant>_<start>_<end>`, so it no longer overwrites the
full-period result. `gradient_boosting_apriori.ipynb` runs steps 1–5 and draws
the figures.

Environment: Python 3.13, packages from `requirements-lock.txt`; 2 vCPU Intel
Xeon 2.30 GHz (Linux), thread pools limited to 2 by the runner. Repository
revision `fd588ef` (main after the Lasso merge) plus this folder's scripts,
bundled snapshot `team/data/`. The January 2023 check matches the lead's
integration table (744 hours, MAE 22.68, 145 / 664 hits).

**Earlier numbers.** The first version of this file used scikit-learn 1.9.1,
which gave a 2023 starter MAE of 18.52 and hit rate of 22.1%, and 44 rules
with cluster labels. With the pinned 1.7.2 the same code gives 18.39, 22.9%
and 46 rules. The model choice and the conclusions below did not change.

## Forecast experiment

Starter: one `HistGradientBoostingRegressor` per local clock hour
(60 iterations, learning rate 0.08, 15 leaves, 30 samples per leaf, L2 = 1).
Each variant changes exactly one setting.

| Run | Change | Year | MAE | RMSE | ±5% hits / eligible | Neg. MAE (n) | Fit s / year |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge (lead) | — | 2023 | 18.78 | 26.65 | 1,575 / 7,738 (20.4%) | 31.56 (281) | 1.2 |
| | | 2024 | 18.34 | 29.52 | 1,356 / 7,620 (17.8%) | 19.43 (375) | 1.2 |
| **GB starter** | — | 2023 | **18.39** | 26.87 | 1,770 / 7,738 (22.9%) | 35.30 (281) | 206 |
| | | 2024 | **16.58** | 27.47 | 1,455 / 7,620 (19.1%) | 20.69 (375) | 208 |
| GB smaller_leaves | `max_leaf_nodes` 15→7 | 2023 | 18.95 | 27.43 | 1,680 (21.7%) | 38.90 | 159 |
| | | 2024 | 17.72 | 28.81 | 1,342 (17.6%) | 25.27 | 159 |
| GB strong_l2 | `l2_regularization` 1→10 | 2023 | 18.66 | 27.24 | 1,734 (22.4%) | 36.87 | 205 |
| | | 2024 | 16.65 | 27.60 | 1,425 (18.7%) | 21.17 | 210 |
| GB absolute_loss | `loss` squared→absolute | 2023 | 19.11 | 27.86 | 1,692 (21.9%) | 39.43 | 226 |
| | | 2024 | 17.32 | 28.97 | 1,577 (20.7%) | 26.20 | 238 |

Hours: 8,760 (2023) and 8,784 (2024) for every run. Fit time is the summed
monthly fit wall time; full-run wall time was 5.4–7.8 minutes per GB run and
4 seconds for Ridge.

**Decision (on the 2023 validation year): keep the starter settings.** None of the three changes
improved 2023 MAE or the 2023 hit rate. Gradient boosting beats Ridge on MAE
in both years (−0.39 and −1.76 EUR/MWh) and on the hit rate, but is worse on
negative-price hours in both years and costs about 175× more fit time.
Smaller trees cut fit time by 23% at a cost of 0.6 (2023) to 1.1 (2024)
EUR/MWh MAE. Absolute-error loss gave the best 2024 hit rate, but it was
worse on 2023 and cannot be chosen from 2024, so this is reported only as an
observation.

## Apriori experiment

Large error = |error| above the 2023 90th percentile of that model's errors
(41.26 EUR/MWh for the GB starter, 40.54 for Ridge). Baskets: season × time
of day × B's K-means cluster. Minimum support 5%, at least 20 large errors in
2023. The large-error rate is 10.0% in 2023 for both models and 6.3% (GB) and
9.0% (Ridge) in 2024; lift is measured against the same year's rate.

Cluster labels: `team/b-random-forest/temporaryresults/regimes.csv` (k = 4,
fitted on days before 2023). Cluster 0 = windy and cloudy, 1 = calm and
clear, 2 = mild and calm, 3 = the day after a very expensive day.

Shipped strongest-rule output for the GB starter:

| Group | Rule | 2023 support / confidence / lift | 2024 support / confidence / lift |
| --- | --- | --- | --- |
| calendar | spring & afternoon | 6.3% / 20.7% / 2.07 | 6.3% / 4.7% / 0.75 |
| cluster | cluster 1 & afternoon | 5.5% / 16.7% / 1.67 | 4.1% / 6.1% / 0.97 |

Neither selected rule holds in 2024. All 46 rules (21 calendar, 25 with a
cluster), from `summarize.py`:

| Rule | GB lift 2023 → 2024 | Ridge lift 2023 → 2024 |
| --- | --- | --- |
| autumn & evening | 2.02 → 2.19 | 2.18 → 1.88 |
| autumn & morning | 1.52 → 1.29 | 1.70 → 1.57 |
| spring & afternoon | 2.07 → 0.75 | 1.99 → 0.63 |
| cluster 1 & afternoon | 1.67 → 0.97 | 0.83 → 0.96 |
| cluster 0 & afternoon | 1.43 → 0.91 | 1.62 → 1.02 |
| cluster 0 & evening | 1.30 → 1.22 | 1.59 → 1.33 |

24 of the 46 GB rules still have lift above 1 in 2024. Four rules have lift
above 1.2 for both models in both years: autumn & evening, autumn & morning,
autumn as a whole, and cluster 0 & evening. Ranking by 2023 lift alone picks
the spring-afternoon rule, which fails in 2024 for both models, and each
model's top cluster rule (cluster 1 & afternoon for GB, cluster 0 & afternoon
for Ridge) also drops to about 1.

**Cluster 3 is invisible to Apriori.** It covers 11 days in 2023 and 18 in
2024 (under 5% of hours), so no itemset containing it reaches the minimum
support. A direct check after the rule search shows it is the one cluster
with a high large-error rate in both years:

| Cluster | Days 2023 / 2024 | GB large errors / hours, 2023 | GB, 2024 | GB MAE 2023 / 2024 | Ridge rate 2023 / 2024 |
| --- | --- | --- | --- | --- | --- |
| 0 | 129 / 131 | 326 / 3,097 (10.5%) | 140 / 3,144 (4.5%) | 19.0 / 14.8 | 12.1% / 8.0% |
| 1 | 80 / 60 | 178 / 1,919 (9.3%) | 73 / 1,440 (5.1%) | 17.4 / 14.8 | 5.0% / 6.0% |
| 2 | 145 / 157 | 326 / 3,480 (9.4%) | 269 / 3,768 (7.1%) | 18.0 / 16.9 | 10.3% / 8.9% |
| 3 | 11 / 18 | 46 / 264 (17.4%) | 69 / 432 (16.0%) | 23.2 / 32.4 | 18.6% / 25.9% |

For GB that is lift 1.74 in 2023 and 2.55 in 2024; for Ridge 1.86 and 2.89.
The Random Forest shows the same pattern (team B). It rests on very few days
and was found by inspection, not by the pre-set rule search.

Caveats: many rules were searched, neighbouring hours are dependent, and
calendar items are coarse (each two-item rule covers 5–11% of hours). No
error correction was attempted.

## Frozen Ridge rules on the boosting forecasts

`python team/c-gradient-boosting/frozen_rules.py` scores the two rules selected
on 2023 Ridge errors on any full-period runs (default: Ridge and the GB
starter), each with its own 2023 90th-percentile threshold, and repeats the
2024 afternoon comparison (cluster 0 vs other clusters, 12:00–17:59) with a
95% interval from 5,000 resamples of contiguous seven-day blocks of delivery
days. On Ridge it reproduces the report's numbers exactly (−2.12 points,
−8.33 to 3.50). Output: `summary/frozen_rules.csv`, `summary/afternoon_comparison.csv`.

| Rule (2024, GB, threshold 41.26) | Events / hours | Rate | Lift |
| --- | ---: | ---: | ---: |
| autumn & evening | 75 / 546 | 13.74% | 2.19 |
| cluster 0 & afternoon | 45 / 786 | 5.73% | 0.91 |

2024 afternoons, GB: cluster 0 5.73% (45/786), all 8.42% (185/2,196), other
clusters 9.93% (140/1,410); difference −4.20 points (−8.43 to 0.44). In 2023
the GB lift of cluster 0 & afternoon (1.43) equals that of afternoon alone
(1.42), so the cluster adds nothing beyond the time band.

## Hand-off files

`temporaryresults/` holds the chosen run (the starter) for the lead:
`predictions.csv.gz` (17,544 aligned hourly forecasts), `metrics.csv`,
`timing.csv` and `run.json`, produced with `requirements-lock.txt`. The
predictions file is temporary and can be removed when merging, as for team A.
`summary/` holds the aggregate tables: `comparison.csv` (all five runs),
`all_rules_<run>.csv` (every rule, 2023 and 2024) and
`apriori_with_clusters.csv` (shipped strongest-rule output),
`frozen_rules.csv` and `afternoon_comparison.csv`.
