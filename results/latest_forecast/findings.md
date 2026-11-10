# Latest-year negative-price forecast experiment

## Goal and data split

Predict two hourly outcomes for connected pairs DK1–DE-LU, DK1–NL and DK2–SE4:
whether at least one area has a negative day-ahead price (`any_negative`), and
whether exactly one does (`discordant`). Every complete hour is included; this
is not conditional on knowing that a negative event occurs. Prices are EUR/MWh;
from the 15-minute market period, the target uses the **hourly mean** price.

As of 10 November 2026 in Copenhagen, the last fully completed local day is
9 November. The test is **10 November 2025–9 November 2026 inclusive**, exactly
365 local days and 8,760 UTC hours per pair. All required input fields and price
targets are present for all test hours. The first 52 days of the window were
already examined in an earlier 2025 experiment. Thus the complete test is not
pristine, though the 2026 segment was not used for this run's model selection.

The final frozen fit uses complete hours from 8 January 2019 to 9 November 2025:
59,952 hours per pair, 179,856 pair-hours total. The first 168 source hours are
excluded because seven-day price lags do not yet exist. These are not independent
examples: the same DK hours appear in two pairs, and weather/prices are serially
correlated. The 365-day test never enters model fitting or standardization.

For development, a fit through 2023 scores 2024 and 1 January–9 November 2025.
The average of the six year/pair Brier scores selects one predefined model per
target; final models are then refit through 9 November 2025. Development dates,
earlier hypotheses, and part of test were already explored, and the feature set
was informed by previous runs; conclusions remain exploratory.

## Inputs and methods

- Copenhagen calendar hour, month, weekday and pair; both areas' previous 48-hour
  and 168-hour day-ahead prices, where the 48-hour lag avoids a same-delivery-day
  price around the autumn DST change.
- Temperature at 2 m, wind speed at 10 m and cloud cover for one representative
  grid point per area. These are JMA GSM archived forecasts through Open-Meteo
  `previous_day2` (fixed 48-hour lead). They are not observations, generation
  forecasts, direct irradiance, or area-wide averages.
- 14-day-lagged EIA Brent spot price in USD/barrel via FRED and its seven-day
  change, for the oil ablation. The 2024–2025 development selection picked a
  model **without** Brent for both outcomes.

The assumed decision time is 10:00 Copenhagen time on the day before delivery.
The 48-hour weather-lead arithmetic is checked against this cutoff. However the
archived API does not provide an individual historical issue/publication time
for every returned value, and the FRED snapshot does not prove historic release
times or original vintages; the two-week oil lag is a conservative assumption,
not a verified point-in-time release feed. Past electricity prices can also be
subject to revisions. This is a prospective-style historical model, not a
fully gate-closure-audited operational backtest.

Fixed candidates are regularized logistic and histogram gradient boosting, with
four feature sets: calendar only, plus lagged electricity prices, plus weather,
and plus Brent. Each pair receives equal total weight during fitting. Separate
pair-specific training-event rates serve as a constant baseline. No features
use contemporaneous realized generation, load, weather, or the target price.
Selection chooses `boosting_weather` for any-negative and `logistic_weather`
for discordance. Parameters, model ranking and assumptions are in `protocol.json`.

## Latest 365-day test

Any-negative prediction, Brier score (smaller is better):

| Pair | Hours | Events | Training-rate baseline | Price-history boost | Selected weather boost | Weather + Brent boost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DK1–DE-LU | 8,760 | 477 | .0519 | .0348 | .0306 | .0319 |
| DK1–NL | 8,760 | 419 | .0458 | .0310 | .0267 | .0276 |
| DK2–SE4 | 8,760 | 141 | .0159 | .0154 | .0144 | .0157 |

Weather lowers Brier relative to the same model with only price history by
.00423, .00428 and .00094, respectively. Seven-day paired-block 95% intervals
are [.00100,.00755], [.00142,.00725] and [.00011,.00189]. These are exploratory
intervals conditional on the fitted models and do not account for multiple model
comparisons, selection or prior period exposure. The SE4 interval is small and
rare-event performance remains uncertain. Brent did not improve the selected
boosting family for any pair in this test, although logistic ablations behave
differently; this does not establish that oil can never be useful.

Average precision of the selected any-negative model is .739, .698 and .195;
event prevalence is .054, .048 and .016. These ranking scores do not establish
calibration. Mean predicted probabilities are .030, .032 and .020 versus
observed rates .054, .048 and .016: the first two pairs underpredict overall.
The model is therefore not ready for uncalibrated operational decision-making.

For the selected discordance classifier, Brier is .0295, .0249 and .0070 versus
the constant baselines .0312, .0259 and .0071. There were 278, 232 and 62 events
respectively. In particular SE4 supports only a limited conclusion. Full log
loss, AUC, average precision, calibration bins, model comparisons, test outcomes
by calendar year, and paired block intervals are in the adjacent CSV files.

The 2026-only portion contains 7,512 hours per pair. Its selected-model Brier is
.0357, .0311 and .0125 against .0603, .0532 and .0142 for the constant baseline;
this is a shorter subperiod of the stated test, not a second independent test.

## Reproduction and retained model

1. Fetch initial inputs using `scripts/fetch_multiarea.py` and
   `scripts/fetch_forecast_inputs.py`; follow `data/multiarea/README.md` and
   `data/external/README.md` for attribution and local-only restrictions.
2. Run `scripts/fetch_latest_forecast_inputs.py` and
   `scripts/train_latest_forecast.py` from the repository root.
3. `scripts/verify_latest_forecast.py` reloads the saved estimators and checks
   every test probability. All 54 target/model/pair groups (26,280 pair-hours per
   model) reproduced within 1e-9. The project test suite passed 38 tests.

`models.joblib` contains the selected and comparison models fitted only through
9 November 2025, and `predictions.csv.gz` contains test results. Both are local
and Git-ignored due to FRED's copyright notice and pending redistribution review.
The selected model has not been refit on the test period and is not a live
production service. For future operational use, obtain run-specific forecast
publication timestamps, point-in-time commodity records, confirm permissions,
evaluate probability recalibration on a separate period, and reserve a new
unseen evaluation window after any model change.
