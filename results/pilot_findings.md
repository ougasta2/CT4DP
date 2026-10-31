# Pilot findings

Scope: complete requested source coverage.
Historical associations in realized settlement data; not causal or day-ahead forecasts.

## Event denominators
| year | PriceArea | n | negative_events | high_events | negative_rate | high_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | DK1 | 8760 | 281 | 438 | 0.0321 | 0.0500 |
| 2023 | DK2 | 8760 | 231 | 438 | 0.0264 | 0.0500 |
| 2024 | DK1 | 8784 | 375 | 226 | 0.0427 | 0.0257 |
| 2024 | DK2 | 8784 | 275 | 256 | 0.0313 | 0.0291 |

## Frozen rules
| event | antecedent | 2023 events/hours | train_lift | 2024 events/hours | test_lift | 2024 interval | test_small_sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| negative_price | area=DK1 & cluster=C0 & season=summer | 54/383 | 4.3953 | 21/507 | 0.9702 | [0.36, 1.92] | False |
| negative_price | area=DK2 & cluster=C1 & season=autumn | 60/763 | 2.9821 | 17/708 | 0.7670 | [0.09, 1.72] | True |
| high_price | area=DK1 & cluster=C3 & season=winter | 204/824 | 4.9515 | 74/919 | 3.1297 | [0.00, 5.54] | False |
| high_price | area=DK2 & cluster=C3 & season=winter | 252/1375 | 3.6655 | 99/1590 | 2.1364 | [0.62, 3.50] | False |

Rules were selected only on 2023, at most one per area and event; exploratory intervals use 400 synchronous seven-day moving-block replicates.
Transfer is assessed by validation lift: >1 indicates enrichment against the matching 2024 area base rate; intervals crossing 1 do not clearly support enrichment.

## Calendar-only comparison
| event | model | n | fallback_rows | brier |
| --- | --- | --- | --- | --- |
| negative_price | calendar | 17568 | 0 | 0.0338 |
| negative_price | calendar_plus_cluster | 17568 | 0 | 0.0321 |
| high_price | calendar | 17568 | 0 | 0.0279 |
| high_price | calendar_plus_cluster | 17568 | 0 | 0.0286 |
Lower holdout Brier score is better. Calendar+cluster uses the same season/time/area groups plus energy cluster; all cell probabilities fit on training only.

high_price: calendar+cluster worsens holdout Brier by 0.000685. This is a descriptive probability baseline, not an operational forecast.
negative_price: calendar+cluster improves holdout Brier by 0.001755. This is a descriptive probability baseline, not an operational forecast.

## Energy regimes (training centroids)
Selected k=4: sampled silhouette 0.353, Davies-Bouldin 0.959, smallest cluster 15.1%, two-seed adjusted Rand agreement 1.000. Silhouette indicates imperfect separation; labels are exploratory summaries.
| cluster | wind_to_load | solar_to_load | load_index |
| --- | --- | --- | --- |
| 0 | 1.0186 | 0.0277 | 1.0142 |
| 1 | 0.2983 | 0.0180 | 0.8091 |
| 2 | 0.3131 | 0.3757 | 1.0158 |
| 3 | 0.3737 | 0.0438 | 1.1739 |
C0: highest wind/load (relative to the other learned centroids).
C1: lowest load index (relative to the other learned centroids).
C2: highest grid-solar/load (relative to the other learned centroids).
C3: highest load index (relative to the other learned centroids).
negative_price, area=DK1 & cluster=C0 & season=summer: point-estimate enrichment does not persist; validation lift 0.97, exploratory interval [0.36, 1.92].
negative_price, area=DK2 & cluster=C1 & season=autumn: point-estimate enrichment does not persist; validation lift 0.77, exploratory interval [0.09, 1.72].
high_price, area=DK1 & cluster=C3 & season=winter: point-estimate enrichment persists; validation lift 3.13, exploratory interval [0.00, 5.54].
high_price, area=DK2 & cluster=C3 & season=winter: point-estimate enrichment persists; validation lift 2.14, exploratory interval [0.62, 3.50].
0/4 selected rules have exploratory intervals entirely above lift 1; point estimates alone do not establish stable enrichment.

Negative-price support: 2 selected rules; small-sample flags apply to 1 (<100 matching hours or <20 matching events in validation). See exact counts above.

## Estimated solar self-consumption sensitivity
| event | antecedent | primary_lift_same_rows | lift | lift_difference | n | events |
| --- | --- | --- | --- | --- | --- | --- |
| negative_price | area=DK1 & cluster=C0 & season=summer | 0.9702 | 0.8622 | -0.1080 | 489.0000 | 18.0000 |
| negative_price | area=DK2 & cluster=C1 & season=autumn | 0.7670 | 0.7724 | 0.0055 | 703.0000 | 17.0000 |
| high_price | area=DK1 & cluster=C3 & season=winter | 3.1297 | 3.1365 | 0.0068 | 917.0000 | 74.0000 |
| high_price | area=DK2 & cluster=C3 & season=winter | 2.1364 | 2.1405 | 0.0040 | 1587.0000 | 99.0000 |
Primary fixed centroids/rules are evaluated with grid solar and total estimated solar on the same available holdout rows. Positive lift_difference means higher enrichment under the estimated total. This is not evidence that the estimate is more accurate.

## Interpretation and limits
Centroids are given in cluster_centers.csv (wind/load, grid-solar/load, area-median load index). Cluster IDs are algorithmic, with no implied ordering.
Gross consumption is a settlement-derived proxy. Grid solar excludes estimated self-consumption. Source metadata warns some reported zeros may represent missing data; zeros are retained as reported, not imputed. This ambiguity cannot be resolved from these sources.
Serial dependence, overlapping rules, training selection and multiple comparisons limit inference. Bootstrap intervals are exploratory and do not correct discovery bias. Repeat with different block lengths before final reporting.
Generation is observed after market clearing. Interconnectors, fuel costs and weather are omitted; no causal interpretation is supported.
Sensitivity results (separately refitted on training only) are in sensitivity/ and sensitivity_comparison.csv; IDs across fits are not equivalent.

## Next iteration
1. Inspect data-quality and solar-sensitivity diagnostics before interpreting rules.
2. Check block-length and rule-selection stability without tuning against 2024.
3. Verify the final submission date on DTU Learn; plan for December 4, 2026 (the assignment also mentions December 6).

Source: Energinet Energi Data Service, CC BY 4.0. Exact cached request envelopes and metadata: data/cache/.
