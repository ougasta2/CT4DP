# Research redesign: spatial discordance of negative prices

## Literature boundary

The former headline—renewable abundance and afternoon hours are associated
with negative prices—is established knowledge and is not a research contribution.
The original wind/solar clustering study remains archived, but its familiar
associations should be background or sanity checks, not the central finding.

1. Grohnheit and Sneum, [Calm before the storm](https://backend.orbit.dtu.dk/ws/files/325898607/1_s2.0_S0301421523002161_main.pdf),
   *Energy Policy* 179 (2023), study DK1 wind, prices, and flexibility for
   2009–2021. A descriptive wind/price relationship is not a gap.
2. [The effect of wind energy production on cross-border electricity pricing](https://www.sciencedirect.com/science/article/abs/pii/S0313592617302011),
   *Economic Analysis and Policy* 58 (2018), already models DK1–DK2 and other
   zone price orderings using wind and planned flows in 2012–2015. Spatial
   price divergence itself is not novel. Only the accessible abstract was
   inspected; detailed methods require full-text review before any novelty claim.
3. Stiewe et al., [Cross-border cannibalization](https://arxiv.org/abs/2405.17166)
   (2024 preprint; published in *Energy Economics*), examine domestic and
   neighbouring renewables across 30 zones in 2015–2023. Regional conditions
   and transmission matter; a two-area domestic proxy cannot isolate them.
4. Rokicki et al., [Predicting Negative Day-Ahead Electricity Prices Across
   12 European Bidding Zones](https://doi.org/10.3390/app16189063), *Applied
   Sciences* 16 (2026), include DK1, evaluate temporal transfer and explicitly
   separate pre-auction variables from later diagnostic information. Negative
   price prediction and model transfer alone are not gaps; realized output
   cannot be presented as an operational forecast.
5. Energinet, [Electricity Market Report 2025](https://en.energinet.dk/media/v2fhaaxv/250226-electricity-market-report-2025.pdf),
   reports simultaneous negative hours, changing east–west price divergence,
   the 29 October 2024 flow-based capacity change, and the 15-minute market
   transition. The old headline and a general increase in divergence are known.
6. The Danish regulator's [2024 national market report](https://www.ceer.eu/wp-content/uploads/2025/09/Denmark_National-Report-2024.pdf)
   already reports 375 and 275 negative hours in DK1 and DK2, respectively.

This targeted search establishes close prior art, not exhaustive proof of
novelty. The specific conditional comparison below is a candidate course-project
contribution, not a claim to be first in the literature.

## Question and design

**Among hours in which at least one Danish bidding zone has a negative hourly
day-ahead price, is the event confined to one zone or shared by both, and does
the realized difference in the two zones' energy regimes add information beyond
their shared regime?** This is a retrospective conditional description; the
condition itself uses the observed outcome. It cannot screen future negative
hours and must not be interpreted causally or as physical congestion.

- Unit: one matched UTC hour with both areas present. Outcome is one if exactly
  one price is negative; zero if both are negative. Hours when neither is negative
  are excluded from this conditional analysis, but their counts are retained.
- Inputs: each area's wind/load, harmonized total-solar/load and load index;
  use their pairwise average and DK1-minus-DK2 difference, plus local calendar.
  Ratios and load are settlement-based proxies and may depend on price, curtailment,
  and accounting. No price, flows, or labels enter the feature matrix.
- Train through 2023, test 2024 and 2025 separately. The load medians in the
  bundled table were also fitted through 2023. 2024 and 2025 were already inspected
  before the question was chosen: all results, including intervals, are exploratory.
- Compare fixed training prevalence, calendar-only logistic regression, calendar
  plus common features, and calendar plus common and differential features.
  Standardization and logistic fitting use training conditional observations only.
  Compare Brier and log loss as well as ranking; report prevalence, counts, and
  a paired seven-day bootstrap. These intervals condition on the fitted models
  and do not correct research selection.
- K-means (k=4, fixed for this feasibility study) uses all training hours and
  six common/difference features without prices. Apriori mines cluster, season
  and time items on training negative hours; choose the highest training lift
  for discordance with at least 5% support, 40 matching hours and 20 discordant
  events. Score that one frozen rule on the test years, including failures.
  Labels are arbitrary and do not measure physical network constraints.
- A 2025 January–September sensitivity avoids the quarter-hour transition; the
  full-year primary target uses the existing hourly means. It does not isolate
  the separate October 2024 market-design change. The bundled 2020 solar estimate
  repairs and pre-2020 definition bridge also remain material limitations.

## Feasibility result: not yet a reliable model

At least one area is negative in 375 hours in 2024 (100 discordant), and 444 in
2025 (182 discordant). The common-plus-difference model has Brier 0.220 in 2024
and 0.255 in 2025. The training-prevalence baseline scores 0.227 and 0.243:
the model is slightly better in 2024 and worse in 2025. Paired intervals for
baseline minus model Brier include zero in both years. The model improves on
the weaker common-only model (0.292 and 0.316), but that is not evidence that
it beats a strong simple baseline. January–September 2025 gives the same
practical conclusion: 0.260 versus 0.242 for the prevalence baseline.

The selected Apriori rule `cluster=C1 & season=spring` does not transfer: its
2024 test lift is 0.31 (2/24) and its 2025 lift is 0.98 (37/92). Do not report
it as a discovered stable pattern. In 2023, 48 DK2-only negative hours occur on
eight UTC dates; temporal dependence and concentration undermine treating hours
as independent replications. The selected clustering and threshold choices
were not subjected to a complete sensitivity study.

See `results/spatial/` for all counts, centers, metrics, intervals, selected
rule and predictions. Reproduce with `.venv/Scripts/python scripts/run_spatial_model.py`.
The approach is a falsifiable prototype, **not an established novel result or a
validated model**. Do not replace failed outcomes with more favorable test-set
choices.

## Requirements for a stronger next version

Use scheduled cross-border flows, available transmission capacity, neighbouring
zone prices or forecast fundamentals only after verifying definitions, historical
coverage, publication time and permissions. Separate a pre-auction forecast from
post-clearing diagnostics; flows and neighbouring clearing prices can be jointly
determined with the target and do not identify causal mechanisms. Freeze a model
and evaluation protocol before an actually untouched future period, include
calibration and a constant-rate baseline, and use rolling-origin development
splits. Investigate how many independent days support each outcome. Revisit the
specific gap after full-text searches for conditional negative-price discordance.
