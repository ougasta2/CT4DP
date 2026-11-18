# Team workflow and experiment protocol

## Question and ownership

How do different model families and their improvements affect the accuracy and computational cost of next-day DK1 hourly price forecasts? This is the proposed team direction, subject to course/TA confirmation. Record the actual allocation and each person's contributions in the final report; do not assume all prior work was jointly produced.

The project lead maintains the existing data and shared experimental design, supplies initial models, investigates ensemble methods, and coordinates the final report. Four other members each take ownership of a model folder, understand its initial implementation, propose and test changes, analyze errors and write up their method and findings. Share responsibility for provenance review and checking the final report.

## Common benchmark

- Target: DK1 day-ahead price, EUR/MWh, hourly average. Since October 2025 the market price is quarter-hourly; the historic pipeline averages it into hourly targets. This is not a 15-minute forecast.
- Intended issue time: 10:00 Europe/Copenhagen on the day before delivery. Inputs are lagged prices for DK1, DK2, DE-LU, NL and SE4, calendar features, and archived JMA weather forecasts. Historic publication and revision vintages have not been established for all observations. This is an exploratory approximation of a 10:00 forecast, not proof of a point-in-time backtest.
- Shared preparation uses `train_dk1_preauction.make_daily` and `train_dk1_weather_ablation.weather_curves`. Training and inference use complete common days. Spring DST's missing model hour is interpolated and the autumn repeated model hour averaged; evaluation retains the actual 23/25 hourly observations, with the same predicted clock-hour value for both repeated hours.
- Expanding training uses only days before the evaluation month; models refit at the start of each month. Evaluation dates must be complete; a missing day fails rather than changing the comparison sample. Training itself can have gaps after input completeness filtering; `timing.csv` records its size.
- Default comparison is 2023–2024. These years have already informed prior experiments and are **development**, not an untouched test. Do not claim independent confirmation from them. Agree with the TA on a feasible later prospective or otherwise independent evaluation before claiming generalization.
- MAE and RMSE use all hours. The ±5% hit rate counts `abs(prediction - actual) <= 0.05 * abs(actual)` only when `abs(actual) >= 20 EUR/MWh`; report hits and eligible counts. Report negative-price MAE and count, plus timing and relevant subgroup errors. Fixed 50% accuracy has not been established.
- `timing.csv` measures fit and prediction wall time per month only, excluding data loading and feature preparation. Compare on the same hardware, environment and thread settings; it is not an end-to-end runtime. The runner limits supported thread pools to two; the forest itself uses one worker.

## Working in parallel

Use the shared CT4DP repository. The member branches and permitted folders are
listed in [team/README.md](../team/README.md). Submit changes to `main` by pull
request; shared protocol or cross-team changes require lead coordination. The
automated scope check is effective only after GitHub branch protection requires
it and blocks direct pushes to `main`.

1. Run the unchanged model and preserve its outputs and version identifier before changing it. `results/team/` is local and ignored; do not commit restricted derivatives.
2. Make one documented hypothesis per experiment, use training-only preprocessing inside `build_model`, and avoid tuning on future or final evaluation observations. Keep the common input rows, forecast horizon and output interface unchanged for the main comparison.
3. Record code revision, data snapshot, dates, hardware, package versions, changes, all attempted settings, outcomes and failures. Compare against the initial model on identical hours. Ask the lead before changing the shared protocol so everyone can rerun it.
4. Deliver model code, a concise explanation, baseline and improved metrics, timing, limitations and report text. MAE, negative-price errors and the hit rate may trade off; no model is automatically best on every measure.
5. The lead aligns the four prediction files, starts with equal weights, and evaluates any weight-learning scheme using chronological out-of-fold predictions. Freeze choices before any genuinely independent final evaluation.

## Known boundaries

Some initial Lasso and MLP fits may issue convergence warnings; record them and investigate before interpreting results. The benchmark is an integration starting point; the four models and the ensemble have not yet been fully run and validated together. Source availability, licensing and individual release timestamps remain open. Realized generation, same-day auction prices and forecasts published after 10:00 are not valid pre-cutoff inputs. In particular, the Energinet day-ahead wind/solar forecast is published at 18:00 on D-1 and belongs only in a separately labeled hindsight experiment. The EXAA 11:00 research question is a separate information horizon and must not be compared as if it were a 10:00 input.

Previous model-selection and bootstrap results are conditional on their own protocols; do not treat uncertainty intervals as correcting for model search. Explain the course's required methods, any method beyond the course, individual contributions and TA feedback in the final report.
