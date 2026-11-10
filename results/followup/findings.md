# Exploratory follow-up: negative-price stability

## Design

For each test year 2020–2025, fit the feature transform, standardization,
K-means selection (k=3–7, silhouette and 3% minimum cluster size), and
predictor-only Apriori rule selection using 2016 through the preceding year.
At most one negative-price rule per area is chosen by training lift, with the
original support and count guards. Evaluate it in the next year; cluster IDs
refer only to that year's fit. 2025 uses 2024 in training, unlike the original
fixed 2016–2023 model. All years have already been explored; these results are
not a pristine confirmation. Realized energy is not a day-ahead predictor.

For a separate, prespecified comparison, the training cluster with the largest
mean solar/load is compared with all other clusters within each test-year,
area, season and time-of-day cell. This descriptive contrast does not control
for remaining differences in wind, load, weather or market conditions. Sparse
cells can be misleading; there are no uncertainty intervals for these contrasts.

## Results

- The training-selected rules change over time. In 2023 the DK1 winter rule
  has test lift 0.38 (11 events / 897 hours); in 2024 the solar-rich afternoon
  rules yield lifts 4.19 (DK1, 215/1,201) and 5.36 (DK2, 175/1,043). Refit
  through 2024, the 2025 rules yield lifts 4.26 (DK1, 262/1,223) and 4.54
  (DK2, 213/1,551). The DK2 2025 rule has no afternoon condition. Selection
  changes and earlier failures qualify any claim of a universally stable rule.
- The 2024–2025 rules above retain lift above one across the reported 3-, 7-
  and 14-day bootstrap intervals. These intervals condition on selected rules,
  do not correct selection or multiple comparisons, and are not independent
  confirmation. Zero observed events may yield a misleading [0, 0] interval;
  missing intervals mean fewer than 80% of replicates had a defined lift.
- Within spring afternoons in 2024, DK1's highest-solar cluster has 84/408
  negative hours versus 9/144 in other clusters; DK2 has 69/342 versus 0/210.
  The contrast varies by season: in DK1 summer afternoons it reverses (115/516
  versus 31/36), with a small comparison group. Thus an area-wide lift is not
  evidence of a uniform advantage within every calendar cell.
- Adding cluster to the calendar-cell baseline improves Brier in both areas in
  2020, 2024 and 2025, but worsens it in both areas in 2021–2023. Differences
  are descriptive and small, and event counts vary substantially across years.
- The previous three-year high-price 95th percentile is an alternative target
  definition, not a replacement for the original fixed threshold. For 2024 it
  is 381.41 EUR/MWh in DK1 and 376.46 in DK2, compared with expanding-window
  thresholds 234.54 and 232.69. The recent thresholds still include crisis year
  2022 and therefore do not solve distribution shift by themselves. Only event
  counts are compared here; no new high-price model is claimed.

## Files and limitations

- `rolling_rules.csv`: training and test denominators, lifts, and block-length
  sensitivities; `rolling_centers.csv` and `rolling_k_diagnostics.csv`: fit details.
- `matched_calendar_cells.csv`: all calendar-stratified counts and rate differences;
  empty comparison groups have undefined rates.
- `rolling_brier.csv`: calendar versus calendar+cluster Brier by area and year.
- `high_price_threshold_sensitivity.csv`: training-only prior-three-year and
  expanding-window thresholds and resulting next-year counts.

The input is the bundled `results/decade/processed.csv.gz`; see the decade
protocol for solar-definition breaks, 2020 repairs, and quarter-hour aggregation.
Existing no-solar and no-repair sensitivities remain in `results/decade/`.
Refitting tests selection stability, but is not a comparison of every possible
training-window length or fixed k. The calendar comparison is associational;
omitted interconnectors, fuel prices and weather prevent causal conclusions.

Source: Energinet Energi Data Service, CC BY 4.0.
