# Extended pilot protocol — November 1, 2026

## Scope and split

Use the latest ten completed calendar years as of the study date: **2016–2025**.
Train on the older eight (2016–2023), test on the newer two (2024–2025), using
Copenhagen local year membership derived from UTC keys. Both DK1 and DK2 are
represented. There is no random train/test split. All expected monthly UTC keys
must exist in each source. At least 99.5% usable coverage per year/area is required;
the actual run retains 100%. Ten years need not mean ten identical measurement
systems: the bridges and sensitivity checks below are part of the study.

The original 2024 period was previously examined. Calling this a pristine new
confirmatory holdout would be misleading; it is exploratory out-of-time testing.
No test performance is used to choose or change model hyperparameters or rules.

## Source compatibility and repairs

Official sources, Energinet Energi Data Service (CC BY 4.0):

- [ProductionConsumptionSettlement](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement)
- [Elspotprices](https://api.energidataservice.dk/meta/dataset/Elspotprices)
- [DayAheadPrices](https://api.energidataservice.dk/meta/dataset/DayAheadPrices)
- [API guide](https://www.energidataservice.dk/guides/api-guides)

The API guide supports bounded requests with explicit record limits (and limit=0
for all rows). This experiment uses bounded windows under 150,000 rows and audits
all expected monthly keys. A seven-year historical request avoids seven separate
five-minute rate-limit waits; successful responses are cached permanently.

Before 2020, all solar is placed in SolarPowerSelfConMWh. From 2020, the harmonized
proxy sums grid solar plus estimated self-use. Pre-2020 grid solar is unavailable,
so its derived grid-only series is marked missing, not claimed to be measured zero.
The raw selected fields remain unchanged in the shared exports.

2020 contains 1,747 negative estimated self-use values (876 DK1, 871 DK2), contrary
to the current metadata's max(modeled-minus-actual,0) definition. Primary features
apply that documented physical bound to this **estimate only**, preserve signed
source values and flag every repair. No missing value is filled with zero. A
strict-no-repair clustering fit excludes all 1,747 affected training rows. Another
fit removes solar entirely. Both keep the original primary event definitions and
test population; the corrected feature is an explicit assumption, not ground truth.

After October 1, 2025, TimeUTC/DayAheadPriceEUR have 15-minute resolution. Validate
unique keys, finite values, quarter alignment and exactly four observations per
UTC hour/area; compute the equal-duration average EUR/MWh. Events refer to this
hourly mean. Source quarter values and the fraction of negative quarters are kept.
Do not sum prices, average incomplete hours or merge on repeated local DST hours.

## Feature fitting and models

The three common clustering inputs are wind/load, harmonized total-solar/load and
load divided by its training area median. Ratios are not clipped. No price input
is used in clustering. All scalers, demand tertiles and the area-specific pooled
eight-year 95th price percentiles are learned from training only.

| Method | Training selection | Holdout handling |
| --- | --- | --- |
| K-means baseline | sklearn k=3..7, 20 initializations, silhouette; reject clusters <3%; two-seed stability | Frozen scaler and centroids |
| New 1: GMM | k=3..7, full covariance, two initializations, BIC; converged solutions with all components >=3% | Frozen mixture predict |
| New 2: DBSCAN | <=20,000 training rows stratified across every year/area; min_samples=20; epsilon is training 20-neighbor-distance quantile .65/.8/.95; prefer silhouette minus .2*noise when >=2 clusters and <=50% noise | Explicit nearest-frozen-core adapter: within epsilon -> core label, otherwise noise -1; never refit on test |
| New 3: Random Forest classification | 120 trees, depth<=12, min_samples_leaf=80, max_features=1.0, seed=42; independent binary targets | Negative/high class probabilities, both full energy+calendar and same-family calendar-only |
| New 4: Histogram gradient boosting regression | 200 iterations, learning_rate=.05, <=15 leaves, min_samples_leaf=100, L2=1, early_stopping=False, seed=42 | Price in EUR/MWh, both energy+calendar and same-family calendar-only |

Supervised calendar inputs are area and cyclic hour/month/weekday. No price-derived
variables, future observations or year identifier are predictors. Hyperparameters
are fixed, not selected on test outcomes. DBSCAN is a bounded pilot approximation;
a single large cluster, noise or undefined sampled silhouette is reported as such.
GMM and K-means fit every training row. CPU thread counts are bounded at two.

## Evaluation and course-method contribution

Classification uses Brier score, average precision, ROC-AUC and fixed-0.5
precision/recall, with counts/base rates. Regression uses MAE, RMSE and mean error
(EUR/MWh). Both include pooled and separate test-year/area scores. Report simple
training calendar cells and same-family calendar-only ablations, not just a best
test score. A paired seven-day moving-block bootstrap compares error reductions;
positive improvement means lower loss. Both areas are resampled synchronously.

Custom predictor-only Apriori is reused for K-means/GMM/DBSCAN cluster baskets.
Support>=2%, max itemset size 3. Select at most one rule per area/event/method with
>=100 antecedent hours and >=20 event hours, ranked by training lift then sample
count. Persist models, preprocessing, configs and selected rules before scoring.
Rare outcome events are independently scored after antecedent mining. Holdout
lift baselines match the price area and evaluation period; also report each test
year separately. Rule intervals use 400 consecutive seven-day block replicates.
Daily sufficient counts implement the same bootstrap as hourly row weights and
are verified against the original implementation.

Multiple comparisons, correlated selected rules, measurement changes, shifting
generation capacity, the 2022 energy-price regime and omitted weather/fuel/exchange
variables limit inference. Paired intervals are exploratory; they do not correct
model-family discovery or establish causality. A future forecasting project needs
pre-clearing predictors and a newly reserved untouched test period.
