# DK1 rolling benchmark: 2023 and 2024

## Objective and evaluation

We use both 2023 and 2024 as requested benchmarks and explicitly allow them to
inform development. Forecast the DK1 day-ahead hourly mean price using data
intended to be available by 10:00 Europe/Copenhagen on the previous day. 2022
monthly walk-forward forecasts provide earlier residual history for calibration;
2023 and 2024 contain 8,760 and 8,784 actual hours. All compared models use
identical days, including weather and fuel completeness; the price data and
weather archives can be revised and their precise historic release timestamps
are not verified. The gas proxy and FRED Brent series are likewise unverified.
The old forecast-weather proxy is an archive at nominal 48-hour lead; it is not
verified that every historical value was available at the cutoff. The 2023/2024
comparisons support development and iteration; using them to select models means
selection bias remains if interpreting their observed best score as an
independent generalization estimate.

Models refit monthly using only previous local days, except that the neural
network refits each January. We compare the original day-curve Ridge;
price-curve extensions; the extensions with 3-day-lagged Gasradar gas proxy and
14-day-lagged Brent; four Lasso/LEAR-inspired models with 8-week, 12-week,
3-year and 4-year windows, with asinh-transformed target; their average; a raw
price 3-year Lasso; an MLP; and an empirical forecast distribution. This is not
an exact implementation of published LEAR: fixed alpha, our features, monthly
refitting and a target transform differ. Fixed hyperparameters, monthly
retraining, asinh inversion and feature engineering should be understood when
reading results. See `protocol.json` and `metrics.csv` for details. Monthly
fits mean hourly observations in each month are not used to refit until the
next month. For day-ahead price labels of prior delivery days we assume auction
results are published normally by the next forecast cutoff; exceptions and
revisions have not been audited.

## Results

±5% means |predicted − actual| ≤ 5% × |actual|, scored only if
|actual| ≥ 20 EUR/MWh. MAE includes every hourly price. It is a hit rate and
not a guarantee that individual prices fall within ±5%.

| Model | 2023 ±5% | 2023 MAE | 2024 ±5% | 2024 MAE |
| --- | ---: | ---: | ---: | ---: |
| Previous day, same hour | 16.93% | 29.98 | 14.92% | 28.90 |
| Original Ridge + weather, monthly | 20.35% | 18.78 | 17.80% | 18.34 |
| LEAR-inspired raw-price Lasso, 3y | 21.50% | 18.29 | 18.07% | 18.15 |
| Same Lasso, past 90 observations per hour median-bias correction | **22.20%** | **18.16** | **18.16%** | **18.01** |
| 50/50 Ridge + raw Lasso blend | 21.13% | 17.97 | 18.41% | 17.79 |
| 4-window asinh ensemble | 16.00% | 21.90 | 17.24% | 19.07 |
| Annual MLP | 13.00% | 25.85 | 13.41% | 20.65 |

The median-corrected raw Lasso was chosen here after inspecting the 2023
candidate table and scored 22.20% (1,718/7,738) in 2023 and 18.16%
(1,384/7,620) in 2024. The Ridge baseline hits 1,575 and 1,356 respectively.
The 2024 improvement is only 0.37 percentage points: it is not evidence of a
large breakthrough. The median correction uses only earlier day forecast
errors, grouped by hour; repeated DST hour errors are averaged before the lag.
The 30/90 observation windows are counts of earlier available same-hour days,
not necessarily 30/90 uninterrupted calendar days. Candidate selection and
both benchmark years are exploratory; see `calibration_metrics.csv` for all
variants, not only the selected row. `paired_intervals.csv` gives conditional
7-day block intervals for raw Lasso versus Ridge: the hit-rate interval includes
zero for each year and is not adjusted for selection.

Fuel ablation with the *extended* Ridge: 2023 ±5% 18.47% without fuel, 18.98%
with fuel; 2024 16.19% without and 15.67% with. MAE changes 19.23→19.10 and
18.43→18.36 EUR/MWh. This does not establish a reliable fuel benefit, nor does
it isolate gas from oil. The original Ridge lacks the extra lags too, so it is
not a pure fuel comparator. See `metrics.csv`.

Forecast distribution from rolling same-hour empirical residuals was tested
for three base forecasts. ±5%-optimized point predictions from raw Lasso show
21.84% and 18.40% for 2023/2024, but MAE worsens to 20.41/19.81 EUR/MWh and
negative-price MAE to 47.03/34.46 EUR/MWh. The 90% interval covers 97.27% in
2023 and 88.27% in 2024, with mean width 140.03 and 69.81 EUR/MWh. These are
not consistently calibrated, and the distribution-based point estimate is not
recommended under the stated MAE/negative-price guardrails. See
`distribution_comparison.csv`.

## Wind/solar forecast audit

Energinet `Forecasts_Hour` snapshots for DK1 and DK2 were downloaded for
2020–2024. The saved official metadata says **ForecastDayAhead is generated at
17:50 and published at 18:00 Danish time on D−1**, after the assumed 10:00
cutoff. `TimestampUTC` refers to ForecastCurrent, not the day-ahead publication
time. Thus `ForecastDayAhead` cannot be used as an actual 10:00 predictor.
The intraday field's detailed historical availability and revisions still
need independent verification; no audited demand, transmission capacity or
EUA histories were added. Historical weather, gas and oil inputs also lack
complete vintage verification.

As an explicit hindsight-only information diagnostic, not a valid pre-auction
forecast, `scripts/train_dk1_renewable_ablation.py` compares common days with
and without this post-cutoff renewable forecast. It excludes missing days
instead of imputing them (2023: 8,736 hours; 2024: 8,664 hours). Raw Lasso
±5% goes 21.75→20.84% in 2023 and 18.26→18.71% in 2024; MAE goes
18.25→18.82 and 18.16→17.36 respectively. This post-cutoff hindsight
comparison is neither a forecast nor a guaranteed upper bound on a feasible
model; it can fail or perform differently because of training and features.
Outputs are in `results/dk1_renewable_ablation/` and the model names explicitly
contain `with_post_cutoff_tso`. The official metadata also states data is
missing 2024-04-12–16. Source: Energinet, Energi Data Service, CC BY 4.0.

## Reproduce and artifacts

- Fetch archived renewable snapshots:
  `.venv/Scripts/python.exe scripts/fetch_dk1_energinet_forecasts.py`
  (obey source rate limits; output includes retrieval URL and checksum).
- Main comparison:
  `.venv/Scripts/python.exe scripts/train_dk1_rolling_benchmark.py`
- Bias and probabilistic comparison:
  `.venv/Scripts/python.exe scripts/evaluate_dk1_calibration.py` and
  `.venv/Scripts/python.exe scripts/evaluate_dk1_distribution.py`
- Hindsight-only TSO information diagnostic:
  `.venv/Scripts/python.exe scripts/train_dk1_renewable_ablation.py`
- Latest available archival-data fit:
  `.venv/Scripts/python.exe scripts/train_dk1_candidate.py`. The Ridge was
  fitted 2019-01-08–2026-11-09 (2,863 days), and 3-year Lasso fitted
  2023-11-11–2026-11-09 (1,095 days). The model file is a fit artifact, not a
  ready-to-run live forecasting service. Fresh inputs and their vintage audit
  are needed; the selected median-bias adjustment is not stored in the latest
  fit because it requires recent rolling out-of-sample residuals.
- Local model and prediction files that incorporate FRED-derived inputs are
  Git-ignored pending redistribution review. The December 2024 stored raw Lasso
  was independently reloaded and reproduced saved December hourly predictions
  to within 5.7e-14 EUR/MWh. Tests: `.venv/Scripts/python.exe -m pytest -q`
  (51 passing; includes DST residual leakage and cutoff vintage tests).
