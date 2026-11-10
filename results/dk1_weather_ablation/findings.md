# DK1 weather ablation: matched price-only and weather models

Reproduce with `.venv/Scripts/python.exe scripts/train_dk1_weather_ablation.py`.
The saved fitted models are in `models.joblib`, predictions for every evaluated
hour are in `predictions.csv.gz`, coverage in `weather_coverage.csv`, and exact
counts and metrics in `metrics.csv`. Existing models and datasets are preserved.

## Experiment

Target: DK1 day-ahead price, EUR/MWh, hourly average. Issue cutoff: previous
local day at 10:00 Copenhagen. Both variants use the same previous-day and
previous-week price curves for DK1, DK2, DE-LU, NL and SE4 and calendar fields.
The weather variant adds 24-hour curves of temperature, wind speed and cloud
cover at one representative site per area (15 weather series total). The same
model settings and identical complete days are used within each paired
comparison. There are 1,819 development-training days, 679 development days,
2,498 final-training days and 365 latest-year test days.

Weather is Open-Meteo's JMA GSM `previous_day2` archive. [Open-Meteo's
specification](https://open-meteo.com/en/docs/previous-runs-api) describes this
as a 48-hour lead series. This is a forecast rather than observed weather, but
its individual historical release/availability and revision timestamps are not
verified. Consequently the cutoff is not fully proven and the result is
exploratory, not an operational point-in-time backtest. This series stitches
fixed lead-time values and is not a single run available at the cutoff.
Representative city wind speed is not wind-generation output, and cloud cover
is not solar-generation output.

Development: train through 2023 and evaluate 2024-01-01 to 2025-11-09.
The model with the best development ±5% hit rate (absolute error <= 5% of
absolute actual; evaluated only when `abs(actual)>=20 EUR/MWh`) is selected;
MAE breaks ties. Refit through 2025-11-09 and score 2025-11-10 to 2026-11-09.
This year has previously been examined and is not an untouched confirmatory
set. MAE covers all hours including negative and near-zero prices. Outputs
remain hourly averages after the 15-minute auction transition; DST 23/25-hour
days are handled as described in `results/dk1_preauction/findings.md`.

## Results

| Model | Development ±5% | Latest-year ±5% | Latest-year MAE EUR/MWh |
| --- | ---: | ---: | ---: |
| Previous day same hour | 16.09% | 18.15% | 32.31 |
| Daily curve, price only | 16.68% | 21.73% | 24.16 |
| Daily curve, price + weather **(selected)** | 19.46% | 24.05% | 20.21 |
| Hourly boosting, price only | 14.28% | 18.83% | 28.01 |
| Hourly boosting, price + weather | 17.61% | 17.39% | 26.20 |

The latest year contains 8,760 hours, of which 7,949 qualify for the relative
metric. The selected daily-curve model hits 1,912 hours within ±5%; price-only
hits 1,727. Average absolute error at the 202 negative-price hours drops from
32.65 to 25.96 EUR/MWh with weather. The absolute-error improvement of weather
for the curve model is 3.95 EUR/MWh and its hit-rate improvement is 2.33
percentage points. A paired seven-day block bootstrap conditional on the fitted
models gives 95% intervals of 3.13–4.85 EUR/MWh and 0.83–3.93 percentage
points respectively; these are exploratory, unadjusted for repeated research
and model selection, and do not fix unverified forecast publication times.
For hourly boosting, weather improves MAE but decreases the ±5% hit rate.
No model achieves ±5% on the majority of eligible hours.

## Missing inputs and next validation

No vetted historical pre-cutoff wind-generation, solar-generation or demand
forecast archive has yet been identified. The Energinet sample audit is stored
in `results/dk1_preauction/energinet_forecast_sample.json`; its record update
timestamp does not establish the original pre-cutoff value. The optional
forecast-vintage input in `scripts/train_dk1_preauction.py` accepts explicit
availability timestamps, but filtering alone cannot prove their provenance or
absence of later revisions. Future work should obtain verifiable vintages,
compare matched training dates with and without them, and evaluate on future
previously unexamined days. Previously used Brent/gas series were not included
in this controlled weather-only ablation because their release/revision times
are not verified either.
