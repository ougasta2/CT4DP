# Multi-area model training and exploratory evaluation

## Question and scope

Conditional on at least one of a connected pair's hourly day-ahead prices being
negative, estimate whether **exactly one** is negative. Eligibility uses the
realized target prices, and features include realized settlement generation and
load; this is retrospective descriptive modelling, not an operational day-ahead
forecast or causal inference. Three pairs: DK1–DE-LU, DK1–NL and DK2–SE4.

Train using 2019–2023, rank the specified candidate families by equally weighted
mean pair-level Brier score on 2024, then evaluate all frozen fits in 2025. The
selected family is pooled logistic regression. A separate historical backtest
trains through 2022 and evaluates 2023. All periods, data and related hypotheses
have had prior exploration; neither 2024 nor 2025 is a clean confirmatory test.
No 2025 score was used in the 2024 ranking for this run.

## Models and features

For each side compute wind/load, solar/load and load relative to its own pair's
training-period median; use the average and signed DK-minus-neighbour difference
of each, plus cyclical local hour/month and weekend. Load medians are fitted on
all usable training hours, then standardized inputs and logistic models are
fitted only on eligible negative-price training hours. C=1, no target-derived
features, and no test observations in the scalers or coefficients. Reported
2025 predictions come from the same 2019–2023 fit as 2024, not a refit on 2024.

- `rate`: each pair's training eligible-event prevalence.
- `calendar`: separate pair logistic model using only calendar.
- `local`: separate pair logistic model using calendar and energy.
- `pooled`: one shared logistic model; training weights make each pair equally
  influential overall.
- `partial_pool`: shared logistic effects plus pair-specific intercept and slope
  deviations with stronger L2 shrinkage (interaction columns scaled by 0.5).
  This is a penalized partial-pooling approximation, not a hierarchical Bayesian fit.
- `loo_pair`: train the shared model on the other two pairs, excluding the
  target pair's outcome labels. Another source may still share a bidding zone.
- `loo_disjoint`: exclude every source pair sharing a bidding zone with the
  target; only one source pair remains when testing a DK1 pair. Both transfer
  experiments use the target pair's **unlabelled 2019–2023 load medians** for
  normalization, so these are not zero-information new-market tests.

Pooled and local models use the same retrospective features; cross-source solar
and demand definitions are not harmonized. No flows, capacity, or neighbouring
market context are included. The three pairs share some locations and common
weather and are not independent datasets.

## Results

Brier score, lower is better. Each row has its own conditional test population.

| Year / pair | Eligible hours | Discordant | Training rate | Calendar | Local | Pooled | Partial pool | Transfer: no shared area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 DK1–DE-LU | 457 | 82 | .187 | .159 | .170 | .142 | .163 | .181 |
| 2024 DK1–NL | 492 | 151 | .291 | .239 | .209 | .251 | .208 | .405 |
| 2024 DK2–SE4 | 645 | 383 | .241 | .208 | .218 | .164 | .199 | .185 |
| 2025 DK1–DE-LU | 583 | 149 | .205 | .182 | .224 | .207 | .218 | .252 |
| 2025 DK1–NL | 604 | 183 | .292 | .245 | .212 | .266 | .213 | .341 |
| 2025 DK2–SE4 | 390 | 141 | .279 | .256 | .234 | .184 | .209 | .185 |

The pooled model was best by equal-pair mean 2024 Brier (.186 vs .199 for local),
but its 2025 equal-pair improvement over local was only .0045, with a paired
seven-day synchronous block-bootstrap 95% interval of [-.0198, .0316]. The
interval is conditional on these fitted models, with no correction for choosing
the model on 2024 or prior exploration. It does not establish a general gain.

Pooling was materially worse on DK1–NL in 2025: .266 versus .212 for local;
local-minus-pooled Brier = -.0538, interval [-.0885, -.0120]. In the no-shared-
area transfer its Brier rose to .341. This is compatible with negative transfer
for this historical test, not proof of the physical mechanism. DK2–SE4 pooling
was better in point estimate, .184 versus .234, but the paired interval for that
gain crosses zero. Pooled DK1–NL ROC-AUC was .372 in 2025; Brier alone does not
establish useful ranking. For DK2–SE4, the local model's mean predicted
discordance probability was .097 against an observed .362: a substantial
calibration problem despite its ROC-AUC of .825.

2025 January–September excludes the price resolution transition: pooled/local
Brier were .198/.220 (DE-LU), .264/.209 (NL), and .195/.218 (SE4). Thus the
NL deterioration is not explained solely by the final quarter's hourly-mean
price definition. This is a period restriction, not a causal test of the change.

## Clustering and association rules

K-means with fixed k=4 is fit on all usable training hours using only the six
energy features, with its own training standardization. For each pair, the
existing independent Apriori implementation mines eligible training baskets of
cluster, season and time; a rule must have at least 5% support, 40 matching
hours and 20 discordant events, then the training rule with maximum lift is
frozen. K and support are fixed design choices; no claim of optimality is made.

The shortlisted rules are too sparse to support general conclusions: in 2025
their matched counts are 6, 0 and 5 hours, respectively. The DE-LU and NL rules
also match just one hour each in 2024. These rules are recorded as failed or
inconclusive transfer, not discoveries. See `rules.csv` for exact denominators.

## Inspect and reproduce

`metrics.csv` has Brier, log loss, AUC, average precision, observed and average
predicted probabilities by pair and year. `calibration.csv` gives fixed-bin
counts and rates. `paired_brier_intervals.csv` gives within-pair comparisons;
`macro_brier_intervals.csv` resamples the same local dates for all pairs and
equally weights pair losses. `sensitivity_2025_jan_sep.csv` holds the restricted
period. Intervals use 1,000 overlapping seven-day blocks, not independent hours.
No interval corrects multiple comparisons or selection. Exact rates and results
can change with data revisions.

Run `.venv/Scripts/python scripts/train_multiarea.py` from the repository root.
`protocol.json` stores model-selection scope and fixed settings. Saved fitted
scikit-learn models and observation-level predictions are local-only and Git-
ignored because SE4 redistribution terms remain unresolved. On this workspace,
reloading the saved models reproduced all seven models' 1,577 conditional 2025
predictions within 1e-10. The complete test suite passed (34 tests).

The raw input provenance, conversion rules, source coverage, pair-hour sample
sizes and limitations are documented in `data/multiarea/README.md`. Data sources:
Energinet Energi Data Service (CC BY 4.0), Energy-Charts / Fraunhofer ISE and
Bundesnetzagentur | SMARD.de (CC BY 4.0 as stated in each response), and eSett
Open Data. eSett redistribution terms remain unresolved.
