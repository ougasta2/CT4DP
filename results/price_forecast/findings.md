# Hourly electricity price regression: latest-year experiment

## Research question and data

Predict hourly day-ahead prices (EUR/MWh) in DK1, DK2, DE-LU, NL and SE4;
compare a previous-local-day same-hour baseline against price history, weather,
and fuel input ablations. This is a point-price model, distinct from the earlier
negative-price classifiers. Each area appears once, rather than duplicating DK1
across pairs. For 15-minute market intervals, both training/test prices are hourly
means; the task is not 15-minute price prediction.

Final fit: 2019-01-08 through 2025-11-09 local time, 59,952 hours per area
(299,760 area-hours). Common weather and interconnected prices mean these are
not independent samples. The first week has no seven-day price history. Test:
2025-11-10 through 2026-11-09, exactly 365 local days / 8,760 hours per area.
Development: fit through 2023 and evaluate each area in 2024 and 2025 through
November 9. Select the model with the lowest equally weighted mean area/year
development MAE, then refit it through November 9, 2025. The test contains 52
days of 2025 previously examined, and the 2026 period was also examined in the
preceding negative-event experiment; this is not a pristine confirmatory sample.

## Inputs and known limitations

- Previous local-day and previous-week same-clock-hour area prices. The previous
  day's auction is assumed available before the notional 10:00 Copenhagen
  cutoff on the prior day. Repeated DST clock hours are averaged and a missing
  spring clock hour falls back to an older UTC observation.
- Forecast temperature, wind speed and cloud cover from JMA GSM's 48-hour
  `previous_day2` archive through Open-Meteo, at one representative city per
  area; this is not an area-weighted wind/solar forecast. The API does not
  provide an individually verified historical publication time for every
  forecast. The lag satisfies the assumed cutoff algebraically, not a full
  real-time vintage audit.
- Brent spot, 14-day lag, EIA via FRED; and a 3-day-lagged EnergyZero-distributed
  gas-day price in EUR/MWh via [Gasradar](https://gasradar.nl/data), which calls
  it a TTF-like wholesale indicator. Gasradar attributes its compilation to
  Gasradar.nl and its source to EnergyZero; the site's compilation states CC BY
  4.0. It is a proxy, not a verified direct historical TTF benchmark. Neither
  fuel series has point-in-time historical release/revision metadata in this
  repository. Short-term price shocks may be missed at these lags.
- Calendar and area indicators plus a German nuclear-shutdown indicator set to
  one from 16 April 2023. Germany's government confirms its final reactors
  stopped on 15 April 2023 ([official source](https://www.bundesregierung.de/breg-de/schwerpunkte/bezahlbare-und-saubere-energie-1581908)).
  All test hours are after this change; the indicator cannot identify its causal
  effect, which is entangled with war, gas prices and many other changes.
  The [ECB](https://www.ecb.europa.eu/press/economic-bulletin/focus/2026/html/ecb.ebbox202606_04~5f026ee8e2.en.html)
  describes 2026 energy shocks and the importance of gas in European wholesale
  power pricing. This experiment does not estimate a war effect.

## Model comparison

Fixed Ridge regression and histogram gradient boosting are compared with feature
sets of price history; +weather; +weather and Brent; +weather and gas; or
+weather, gas and Brent. Training gives each area equal total weight. The 2024–
2025 development winner is **boosting + weather**, with mean development MAE
22.29 EUR/MWh; +weather+gas is 22.32 and is not selected. No 2026 test score
is used to choose the winner in this run.

Latest-year MAE in EUR/MWh and ±3% accuracy, where percentage accuracy is
computed only for hours with `abs(actual price) >= 20 EUR/MWh`:

| Area | Previous-day MAE | Selected MAE | Selected eligible hours | Selected ±3% | Previous-day ±3% |
| --- | ---: | ---: | ---: | ---: | ---: |
| DE-LU | 31.38 | 25.08 | 7,933 | 12.6% | 12.6% |
| DK1 | 32.31 | 24.73 | 7,949 | 11.7% | 10.8% |
| DK2 | 34.44 | 29.16 | 7,939 | 7.9% | 10.6% |
| NL | 29.49 | 24.05 | 8,031 | 12.8% | 13.7% |
| SE4 | 37.28 | 29.68 | 7,404 | 6.9% | 8.7% |

MAE is lower for all five areas, but ±3% remains uncommon and the naive baseline
is at least as good on that threshold in four areas. MAE was the development
selection target, not the ±3% rate. The percentage denominators exclude the
near-zero prices where relative errors are unstable; the cutoff is a reporting
choice, not a market threshold. Each reported MAE includes every one of the
8,760 hours, including negatives and near-zero prices.

Compared with the same boosted price-history model, the weather model improves
test MAE by 3.97, 5.38, 2.75, 3.26 and 4.33 EUR/MWh (DE-LU, DK1, DK2, NL, SE4).
Paired seven-day block 95% intervals for these differences are wholly above zero;
they are conditional on the fitted models and do not adjust for model selection,
multiple tests or prior exploration. Adding gas to weather shows small point
improvements in four areas and a worsening in SE4; all five paired intervals
cross zero. Brent likewise does not show a consistent improvement. The 2026-only
test segment has 7,512 hours per area; its selected MAEs are 25.79, 25.65,
30.09, 25.63 and 30.16, respectively (`test_by_calendar_year.csv`).

## Limits: extremes, negative sign and gas regime

Selected-model MAE for prices >=200 EUR/MWh is 51.64, 59.32, 67.91, 53.88 and
67.19 EUR/MWh respectively, showing a large tail error. MAE at negative prices
is 39.49, 23.60, 17.61, 38.77 and 22.55 EUR/MWh; all price bands and counts are
in `price_band_metrics.csv`. A price prediction below zero detects only 117/474,
54/202, 19/94, 109/404 and 37/126 actual negative hours (24.7%, 26.7%, 20.2%,
27.0%, 29.4% recall). The separate earlier negative-event classifier is not the
same as taking the sign of this regression.

The training-period 75th percentile of the lagged gas proxy is 47.09 EUR/MWh.
For DK1, test hours above this threshold have selected-model MAE 29.92 EUR/MWh
(4,440 hours), versus 19.39 for the remaining 4,320. This is descriptive and
may reflect season, demand, price levels or other simultaneous market changes;
it is not an estimated causal impact of gas prices or geopolitical events.

## K-means and Apriori error diagnostics

K-means (k=4 fixed) is fit on 2019–2023 lagged prices and weather inputs only,
without current target prices. After the model is selected, development-period
absolute errors define an area-specific 90th-percentile large-error threshold.
The independent Apriori implementation selects one cluster/calendar antecedent
per area on development data; it is evaluated with its threshold fixed on test.
For example, DK1's selected autumn/cluster rule matches 763 test hours and 302
large errors (39.6%, vs 13.2% overall). This is an exploratory, selected and
unadjusted comparison; temporal dependence, prior exploration and differing
calendar composition prevent a causal or confirmatory interpretation. Other
rules, including a DK2 rule with test lift approximately 1, are retained in
`error_rules.csv`. A low MAE alone hides these failures.

## Inspect and reproduce

`test_metrics.csv`, `price_band_metrics.csv`, `gas_regime_metrics.csv`,
`negative_sign_metrics.csv`, `mae_intervals.csv`, `test_by_calendar_year.csv`,
`error_rules.csv`, `cluster_centers.csv` and `protocol.json` record the results.
Use `scripts/fetch_price_inputs.py` for the gas input, the earlier documented
fetchers for historical/latest electricity, weather and Brent, and run
`scripts/train_price_regression.py`. The selected and comparator fitted models
and 43,800 test zone-hour predictions per model are stored locally as
`models.joblib` and `predictions.csv.gz`; they are Git-ignored due to FRED
copyright / redistribution review. The selected model does not itself use fuel,
but the shared estimator file includes oil-dependent models.

`scripts/verify_price_regression.py` reloads all models and reproduced all 55
model/area prediction groups within 1e-9. The full suite passed 40 tests. This
is a reproducible research prototype, not a live service; a reliable operational
version needs per-run forecast and commodity release/vintage timestamps,
out-of-sample calibration and an actually untouched future evaluation period.
