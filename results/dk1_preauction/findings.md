# DK1 pre-auction experiment

## Scope and reproducibility

Run `.venv/Scripts/python.exe scripts/train_dk1_preauction.py`. The task is the
DK1 next-local-day hourly mean day-ahead price in EUR/MWh, issued at 10:00
Europe/Copenhagen on the previous day. After the 15-minute market transition
this predicts the average of four prices, not each quarter-hour price. The input
contains the full previous-local-day and previous-week same-clock price curves
for DK1, DK2, DE-LU, NL and SE4 and calendar information. Day-ahead results
for the previous delivery day are assumed available by the cutoff; exceptional
publication delays and archive revisions have not been verified. Weather,
Brent, gas and TSO forecasts are not included in this run because their
historical publication and revision times have not been established.

The ridge model outputs 24 clock-hour prices together, using the same input as
an hourly histogram gradient boosting model. On the autumn DST day repeated
clock-hour prices are averaged for curve training and receive the same
prediction; both actual hours are scored. On the spring DST day the absent
clock hour is interpolated for training targets and lag inputs and not scored.
Prices are hourly averages, including after the market transition. There is no
claim of predicting individual 15-minute intervals.

Train through 2023 and evaluate 2024-01-01 to 2025-11-09 for model selection.
Rank the naive baseline and both models by ±5% hit rate on hours with
`abs(actual)>=20 EUR/MWh` (MAE breaks ties). Refit through 2025-11-09 and score
2025-11-10 through 2026-11-09. The test year was previously inspected, so this
is exploratory and future untouched data is needed for confirmation.

## Results

| Model | Development ±5% | Latest-year ±5% | Latest-year MAE EUR/MWh |
| --- | ---: | ---: | ---: |
| Previous day, same hour | 16.09% | 18.15% | 32.31 |
| Ridge, 24-hour curve (selected) | 16.68% | 21.73% | 24.16 |
| Hourly gradient boosting | 14.28% | 18.83% | 28.01 |

Latest-year ±5% denominator is 7,949 hours, excluding near-zero actual prices;
MAE includes all 8,760 hours. Ridge has 1,727 hits against 1,443 for the naive
baseline. This is a hit rate, not a promise of errors within ±5%. The selected
model's MAE on 202 negative-price hours is 32.65 EUR/MWh, versus 25.84 for the
naive baseline. The pre-existing weather-based model reported 18.87% and 24.73
EUR/MWh for DK1 in this same explored year; its weather issue timestamps were
not verified and its training pooled five zones, so this comparison combines
several changes and cannot identify their individual effects.

## Forecast vintage audit and input contract

A small official Energinet `Forecasts_Hour` API response is saved as
`energinet_forecast_sample.json`, with URL, retrieval timestamp and SHA-256.
It contains `ForecastDayAhead` but also a `TimestampUTC` around delivery time;
this sample does not prove what was actually published before the previous-day
10:00 cutoff or whether values were subsequently revised. It is not used for
training. See [Energinet's API guide](https://www.energidataservice.dk/guides/api-guides)
and [dataset](https://www.energidataservice.dk/tso-electricity/Forecasts_Hour).
A wider source/vintage audit is still required to establish coverage and obtain
historical forecasts suitable for a real backtest.

For a vetted, retained historical-vintage CSV, use
`python scripts/train_dk1_preauction.py --forecasts PATH`. Columns are
`target_utc,available_utc,source,variable,value`; both timestamps must include
UTC offsets. For each target/source/variable, the latest record available by
the previous local day at 10:00 is selected. Complete hourly coverage is
required for every development and test day, including both autumn DST hours;
missing training days are excluded. Outputs go to
`results/dk1_preauction_with_forecasts/`. To compare the impact of a forecast
source fairly, rerun price-only models on the exact same training/evaluation
days; the current separate runs are directly comparable only if their date
coverage is identical. Filtering timestamps cannot establish that timestamps
are truthful or that the archive has not been revised: the source must support
those claims independently. No validated forecast file is supplied here.

`metrics.csv`, `predictions.csv.gz`, `models.joblib`, and `protocol.json` are
local experimental artifacts. Tests cover vintage cutoff and DST leakage;
the suite is run with `.venv/Scripts/python.exe -m pytest -q`.
