# DK1 pre-auction forecasting: research status

The objective is to predict the next day's hourly mean DK1 wholesale price,
including a forecast of electricity demand made before bidding. The primary
score is the fraction of hours within 5% of the actual price, evaluated only
when the absolute actual price is at least EUR 20/MWh. MAE includes all hours.
For prices after the market's 15-minute transition, the target is the mean of
the four quarter-hour prices. The requested 50% hit rate has **not** been
established on a robust, independent evaluation.

## Demand and 10:00 forecast

`train_dk1_demand_forecast.py` fits a monthly 24-output ridge model with
14/21/28-day lagged consumption curves, weather forecasts, and calendar terms.
Its training labels precede the forecast month by at least 10 days. Its demand
predictions are generated out of fold before they enter the price models.
This is an adaptation of the load-forecasting regression family discussed by
[Hong et al.](https://doi.org/10.1016/j.ijforecast.2013.07.001), not an exact
reproduction. `train_dk1_vanilla_load.py` separately implements the calendar,
temperature polynomial, and interactions of [Ziel and Liu](https://doi.org/10.1016/j.ijforecast.2016.01.001),
using ridge regularization.

The 2026 January–October backtest of the nonlinear 10:00 model has 1,617 hits
in 6,504 eligible hours (24.86%); all-hour MAE is EUR 20.18/MWh. A lasso
alternative achieved 25.74% and MAE EUR 20.79/MWh. These results are exploratory:
2026 data have already been inspected, and source publication/revision times
have not been audited for every individual weather, fuel and settlement value.
See `evaluation_2026_metrics.csv` and `evaluation_2026_protocol.json`.

## Early-auction information at 11:00

The separate 11:00 experiment assumes the German EXAA 10:15 auction result is
available before issue time. Exact historical publication times and revisions
have not been verified. The source coverage is incomplete: results apply to
hours with four EUR-labelled early-auction quarter prices for both AT and DE.
Raw German EXAA gave 53.13% in August 2026 and 40.44% in September–October.
August was used to select among several corrections; the subsequent period had
also been examined earlier and is not a pristine holdout.

The fixed candidate `EXAA_DE + 0.1 * (DK1_previous_day - DE_previous_day)`
was selected using August only, subject to no worse MAE than raw EXAA. In
September–October it achieved 557/1,333 = 41.79% versus 539/1,333 = 40.44%
for raw EXAA, with all-hour MAE EUR 16.48 versus EUR 16.94/MWh. A paired
seven-day block bootstrap gives a conditional 95% interval of 0.45–2.44
percentage points for the hit-rate difference. This interval does not correct
for the search over models or the earlier examination of the test period.
See `../dk1_exaa_de/spread_protocol.json`, `spread_test_metrics.csv` and
`spread_test_paired_interval.csv` in that directory.

An additional paired residual-model ablation adds both directions of the DK1–DE
day-ahead net transfer capacity from an ENTSO-E-backed historical download.
On the same September–October hours, the hit rate was 38.03% with capacity
versus 38.48% without it; MAE was EUR 16.45 versus EUR 16.47/MWh. This does
not establish an improvement. The capacity backfill can contain later revisions
and therefore cannot yet be treated as a verified 11:00 input. See
`../dk1_exaa_de/capacity_protocol.json` and `capacity_metrics.csv`.

The full Austrian EXAA history was subsequently obtained from APG for every
day in 2023–2025. In `../dk1_exaa_full/`, an 11:00 stacked model selected using
the full 2024 data achieved 24.86% (1,894/7,620) in 2024 and 31.05%
(2,302/7,415) in 2025. Its 2025 all-hour MAE was EUR 13.49/MWh, compared with
27.05% and EUR 16.21/MWh for the 10:00 baseline. The paired seven-day bootstrap
interval for the 2025 hit-rate improvement is 2.27–5.73 percentage points,
conditional on the chosen model; it does not correct for earlier experiments.
All 8,760 hours of 2025 are included in the MAE, and all 7,415 eligible hours
in the hit rate. 2025 has been viewed before and is not a pristine holdout.

The full-history Austrian capacity ablation uses matched hours. With capacity
it achieved 25.77% in 2024 and 31.03% in 2025, compared with 24.88% and 30.69%
without capacity. The corresponding 2025 paired bootstrap interval for the
hit-rate difference is −0.15 to 0.93 percentage points; the later-revised
capacity history prevents a firm live-use claim. The Austrian early auction is
distinct from the stronger but much shorter German EXAA series. The earlier
one-in-four-day results in `../dk1_exaa_sample/` are samples, not annual results.
The October 2026 fit of the selected Austrian stack is saved locally as
`../dk1_exaa_full/october_2026_model.joblib`, with its feature order and last
training target date; the training script checks predictions after reload.
It is a historical research artifact and requires the upstream demand, price,
weather, and EXAA feature pipeline, not a standalone live forecast service.

The day-ahead ENTSO-E load forecast backfill is a revised historical series;
it is retained for comparison and excluded from model selection. Exact vintage
and release-time evidence is still needed before claiming a faithful live
backtest. No result above demonstrates the requested 50% target.
