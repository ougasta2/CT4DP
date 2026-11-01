# Ten-year pilot findings



**Training: 2016–2023 (eight years). Test: 2024–2025 (two later years).**

The latest ten completed calendar years as of November 1, 2026 are 2016–2025. Source coverage is checked before fitting; no random train/test split is used.



Usable training area-hours: 140,256; test area-hours: 35,088.



## Main reading



Random Forest negative-event Brier improves from 0.036060 (calendar only) to 0.032687 (energy + calendar), a 9.4% relative reduction. High-event Brier worsens from 0.012078 to 0.022764. The paired intervals below quantify those changes.

Gradient-boosting MAE changes from 35.65 to 41.12 EUR/MWh when energy features are added. More realized-energy inputs do not uniformly improve out-of-time performance.

The fixed K-means afternoon/solar-rich negative-price rules have pooled test lifts area=DK1: 4.24 (492/2494 events/hours), area=DK2: 5.52 (362/2132 events/hours). See rule_by_test_year.csv for each year separately.

88.4% of training high-price events occur in 2022. Their pooled test prevalence is only 0.93%, versus about 5% in pooled training by definition. This documents a substantial target-distribution shift; it does not identify a causal explanation.

All methods use realized settlement generation/load. Classifiers and regressors are retrospective conditional models, **not operational day-ahead forecasts**. The 2024 period was already examined in the original pilot, so this is exploratory temporal evaluation, not an untouched confirmatory holdout.



## Data compatibility



Before 2020, the source places all solar in SolarPowerSelfConMWh. From 2020, total reported solar is grid components plus estimated self-consumption. The decade uses this **reported total solar proxy**, not a fictitious pre-2020 grid-solar series. Metadata warns of a measurement/layout break and uncertain gross-consumption accounting. See the separate wind/load-only sensitivity and annual profiles.

The 2020 data contain 1,747 negative estimated self-use values, despite the metadata definition max(modeled production minus actual, 0). The harmonized feature applies that stated nonnegative bound, preserves the original signed values and flags all repairs. This is a documented repair assumption, not zero-filling missing observations. A separate strict clustering sensitivity excludes every repaired row and still represents all eight training years; event definitions and the test population stay fixed.

From October 1, 2025, four complete 15-minute prices are averaged to EUR/MWh per hour. Negative/high events then concern the hourly mean; a negative quarter need not make its hourly mean negative. Exact quarter counts, original responses and the transition audit are retained. Prices are not summed.



## Event counts and fixed thresholds



| year | PriceArea | n | negative_events | high_events | negative_rate | high_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | DK1 | 8784 | 62 | 0 | 0.0071 | 0.0000 |
| 2016 | DK2 | 8784 | 49 | 0 | 0.0056 | 0.0000 |
| 2017 | DK1 | 8760 | 84 | 0 | 0.0096 | 0.0000 |
| 2017 | DK2 | 8760 | 58 | 0 | 0.0066 | 0.0000 |
| 2018 | DK1 | 8760 | 51 | 0 | 0.0058 | 0.0000 |
| 2018 | DK2 | 8760 | 40 | 1 | 0.0046 | 0.0001 |
| 2019 | DK1 | 8760 | 132 | 0 | 0.0151 | 0.0000 |
| 2019 | DK2 | 8760 | 95 | 0 | 0.0108 | 0.0000 |
| 2020 | DK1 | 8784 | 192 | 0 | 0.0219 | 0.0000 |
| 2020 | DK2 | 8784 | 89 | 2 | 0.0101 | 0.0002 |
| 2021 | DK1 | 8760 | 87 | 337 | 0.0099 | 0.0385 |
| 2021 | DK2 | 8760 | 24 | 388 | 0.0027 | 0.0443 |
| 2022 | DK1 | 8760 | 38 | 3133 | 0.0043 | 0.3576 |
| 2022 | DK2 | 8760 | 5 | 3070 | 0.0006 | 0.3505 |
| 2023 | DK1 | 8760 | 281 | 37 | 0.0321 | 0.0042 |
| 2023 | DK2 | 8760 | 231 | 46 | 0.0264 | 0.0053 |
| 2024 | DK1 | 8784 | 375 | 67 | 0.0427 | 0.0076 |
| 2024 | DK2 | 8784 | 275 | 78 | 0.0313 | 0.0089 |
| 2025 | DK1 | 8760 | 441 | 84 | 0.0503 | 0.0096 |
| 2025 | DK2 | 8760 | 265 | 98 | 0.0303 | 0.0112 |



High-price thresholds are the area-specific pooled 2016–2023 95th percentiles, held fixed for both test years. This differs from the original 2023-only threshold.



## Unsupervised methods



| method | split | n | clusters | noise_fraction | silhouette | davies_bouldin |
| --- | --- | --- | --- | --- | --- | --- |
| kmeans | train | 140256 | 4 | 0.0000 | 0.3557 | 0.9177 |
| kmeans | test | 35088 | 4 | 0.0000 | 0.3107 | 0.9836 |
| gmm | train | 140256 | 7 | 0.0000 | 0.0239 | 2.6392 |
| gmm | test | 35088 | 7 | 0.0000 | 0.0295 | 3.9298 |
| dbscan | train | 140256 | 9 | 0.2628 | -0.1153 | 1.2577 |
| dbscan | test | 35088 | 9 | 0.4582 | -0.3004 | 1.4135 |

K-means uses k=3..7, training-only standardization, silhouette and a 3% smallest-cluster guard. GMM selects among 3..7 full-covariance mixtures by training BIC, convergence and the same size guard. DBSCAN fits a deterministic <=20,000-row year/area-stratified training sample; every training/test row is assigned via frozen core neighborhoods. This is a bounded pilot approximation, not a native DBSCAN predict operation. Noise remains -1.



## Classification: temporal holdout



| method | features | event | n | events | brier | average_precision | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calendar_cells | calendar_only | negative_price | 35088 | 1356 | 0.0373 | 0.1015 | 0.7650 |
| kmeans | calendar+cluster | negative_price | 35088 | 1356 | 0.0367 | 0.1056 | 0.7736 |
| gmm | calendar+cluster | negative_price | 35088 | 1356 | 0.0360 | 0.1709 | 0.8186 |
| dbscan | calendar+cluster | negative_price | 35088 | 1356 | 0.0372 | 0.1365 | 0.8173 |
| calendar_cells | calendar_only | high_price | 35088 | 327 | 0.0112 | 0.0129 | 0.6194 |
| kmeans | calendar+cluster | high_price | 35088 | 327 | 0.0143 | 0.0119 | 0.6261 |
| gmm | calendar+cluster | high_price | 35088 | 327 | 0.0139 | 0.0132 | 0.6453 |
| dbscan | calendar+cluster | high_price | 35088 | 327 | 0.0118 | 0.0124 | 0.5887 |
| random_forest | calendar_only | negative_price | 35088 | 1356 | 0.0361 | 0.1931 | 0.7853 |
| random_forest | energy_calendar | negative_price | 35088 | 1356 | 0.0327 | 0.2230 | 0.8321 |
| random_forest | calendar_only | high_price | 35088 | 327 | 0.0121 | 0.0415 | 0.7751 |
| random_forest | energy_calendar | high_price | 35088 | 327 | 0.0228 | 0.0173 | 0.7015 |

Lower Brier is better. Average precision should be read against the event prevalence; ROC-AUC alone can hide rare-event weakness. Full per-year/area results and fixed-0.5 precision/recall are in classification_metrics.csv. Cluster probability cells and all calendar backoffs use training outcomes only.



## Price regression: temporal holdout



| method | features | n | mae_eur_mwh | rmse_eur_mwh | mean_error_eur_mwh |
| --- | --- | --- | --- | --- | --- |
| calendar_cells | calendar_only | 35088 | 47.5817 | 61.5823 | -35.3181 |
| gradient_boosting | calendar_only | 35088 | 35.6507 | 48.6233 | -6.9054 |
| gradient_boosting | energy_calendar | 35088 | 41.1199 | 57.0546 | 18.2477 |

MAE/RMSE are EUR/MWh; smaller is better. HistGradientBoostingRegressor uses fixed hyperparameters and early_stopping=False, avoiding random internal validation. No holdout target is used in fitting or selection.



## Paired energy-versus-calendar comparisons



| method | target | loss | improvement | ci_low | ci_high | repeats | block_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | negative_price | Brier | 0.0034 | 0.0019 | 0.0051 | 400 | 7 |
| random_forest | high_price | Brier | -0.0107 | -0.0129 | -0.0088 | 400 | 7 |
| gradient_boosting | SpotPriceEUR | MAE EUR/MWh | -5.4693 | -8.4294 | -2.2238 | 400 | 7 |

Positive improvement means adding realized energy features lowers loss against the same model family using calendar features alone. Intervals use paired seven-day moving blocks of the test period, preserving both areas and serial dependence. These exploratory intervals do not account for trying multiple model families.



## Solar-definition and correction sensitivity



| method | event | n | events | brier | average_precision |
| --- | --- | --- | --- | --- | --- |
| kmeans_without_solar | negative_price | 35088 | 1356 | 0.0376 | 0.0797 |
| kmeans_without_solar | high_price | 35088 | 327 | 0.0116 | 0.0171 |
| kmeans_strict_no_solar_repair | negative_price | 35088 | 1356 | 0.0367 | 0.1060 |
| kmeans_strict_no_solar_repair | high_price | 35088 | 327 | 0.0143 | 0.0120 |

The no-solar variant removes the measurement bridge entirely. The strict-no-repair variant retains only original nonnegative estimates in training, using the fixed primary event definitions and the same two test years. Differences are reported, not used to retune the primary models.



## Apriori rules for the three clustering methods



| method | event | antecedent | train_lift | test_events/hours | test_lift | ci_low | ci_high | test_small_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmeans | negative_price | area=DK1 & cluster=C2 & time_of_day=afternoon | 3.7183 | 492/2494 | 4.2414 | 3.6341 | 4.9759 | False |
| kmeans | negative_price | area=DK2 & cluster=C2 & time_of_day=afternoon | 3.8372 | 362/2132 | 5.5164 | 4.7007 | 6.4640 | False |
| kmeans | high_price | area=DK1 & cluster=C2 & season=summer | 2.8637 | 1/2008 | 0.0579 | 0.0000 | 0.2162 | True |
| kmeans | high_price | area=DK2 & cluster=C2 | 2.0703 | 0/3460 | 0.0000 | 0.0000 | 0.0000 | True |
| gmm | negative_price | area=DK1 & cluster=C5 & time_of_day=afternoon | 4.6861 | 500/2506 | 4.2897 | 3.7133 | 5.0216 | False |
| gmm | negative_price | area=DK2 & cluster=C0 & demand=low | 4.2296 | 18/1336 | 0.4377 | 0.1024 | 0.8454 | True |
| gmm | high_price | area=DK1 & cluster=C5 | 1.8667 | 1/4445 | 0.0261 | 0.0000 | 0.0954 | True |
| gmm | high_price | area=DK2 & cluster=C6 & season=summer | 1.6972 | 1/700 | 0.1424 | 0.0000 | 0.5293 | True |
| dbscan | negative_price | area=DK1 & cluster=C-1 & demand=mid | 2.4515 | 37/1566 | 0.5080 | 0.2656 | 0.7665 | False |
| dbscan | negative_price | area=DK2 & cluster=C-1 & demand=mid | 2.8148 | 178/1449 | 3.9910 | 2.9561 | 5.2427 | False |
| dbscan | high_price | area=DK1 & cluster=C-1 & season=summer | 1.5961 | 2/2766 | 0.0840 | 0.0000 | 0.2192 | True |
| dbscan | high_price | area=DK2 & cluster=C-1 & season=summer | 1.7313 | 0/2370 | 0.0000 | 0.0000 | 0.0000 | True |

Predictor-only Apriori retains rare outcomes. At most one training-selected rule per area/event/method; >=2% antecedent support, >=100 hours and >=20 events. Rules are frozen before holdout scoring. Year-specific counts/lifts are in rule_by_test_year.csv; do not treat pooled two-year enrichment as stability in each year.

A rule with zero observed test events can have a degenerate [0,0] empirical bootstrap interval. This does not show that its future event probability is zero; small-sample flags and exact counts take priority over an apparently narrow interval.



## Limits and next iteration



Ten years span major market and measurement changes. Pooled thresholds, capacity growth and distribution drift can change event prevalence; no installed-capacity normalization, weather, fuel or interconnector features are added. No causal claim is supported. Cluster IDs are method-specific. DBSCAN separation failure or high noise is a result, not a reason to tune against the test set.

1. Inspect 2020 solar-definition and 2025 interval-transition sensitivity before substantive interpretation.

2. Check year/area scores and paired intervals rather than selecting the best pooled test score for retuning.

3. If building a forecasting study later, obtain genuinely pre-clearing predictors and reserve a new untouched period.



Source: Energinet Energi Data Service, CC BY 4.0. Official metadata, raw selected-field responses and derived data are bundled with the study.
