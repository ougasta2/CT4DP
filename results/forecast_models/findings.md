# Archived weather and Brent: exploratory all-hour prediction

## Research basis

- [Lago et al., forecasting benchmark and review](https://arxiv.org/abs/2008.08004)
  stress realistic market-time inputs, strong benchmarks and sufficiently long
  out-of-time comparisons; their work concerns price levels, not this binary target.
- [Probabilistic forecasts of load, solar and wind](https://arxiv.org/abs/2501.06180)
  finds value in using forecasts of power-system fundamentals. Here the weather
  variables are point-location proxies, not generation or demand forecasts.
- [Danish day-ahead price forecasting](https://www.mdpi.com/1996-1073/12/5/928)
  considers fossil-fuel and weather variables; its reported results do not imply
  every weather variable helps. An accessible abstract/summary was inspected.
- [Weather and commodity impacts on day-ahead electricity](https://www.mdpi.com/2673-4826/7/1/10)
  examines weather and gas, coal and oil jointly. It is not evidence that crude
  oil should improve this particular short-horizon event classifier.

These references motivate ablation tests; none guarantees a transferable gain.
European gas, carbon prices, load forecasts and renewable forecasts are absent.

## Outcomes and information timing

Unlike the previous conditional-on-an-event analysis, this study uses **all
hours** with complete inputs, and trains separate binary classifiers for:

1. `any_negative`: at least one area's hourly-average day-ahead price is <0.
2. `discordant`: exactly one area's hourly-average price is <0.

Each is a price-event classifier, not a price-level regression or a forecast of
the direction of a discordance. Model inputs contain no contemporaneous observed
generation, load or price. Train 2019–2023, use 2024 to compare fixed candidate
specifications, and evaluate frozen models on 2025. The first 168 hours of each
2019 pair are unavailable due to seven-day price lags; all hours in 2024 and 2025
have complete selected inputs. Both years and related hypotheses were already
explored; this is not independent confirmation.

The notional decision time is 10:00 Europe/Copenhagen on the previous local day.
Inputs are calendar, pair indicators and 48/168-hour lagged day-ahead prices;
the 48-hour lag is used to avoid accessing same-delivery-day clearing prices
around the autumn DST transition. Wind speed at 10 m, temperature at 2 m, and
cloud cover use one JMA GSM grid point per area through [Open-Meteo's previous
runs API](https://open-meteo.com/en/docs/previous-runs-api), with its
`previous_day2` 48-hour fixed lead. Cloud cover is a solar-related proxy, not
irradiance or solar generation. Points are Esbjerg, Copenhagen, Hamburg, Amsterdam
and Malmö, not area-wide or offshore averages.

The fixed 48-hour lead is conservatively earlier than the assumed cutoff for all
delivery hours, and the code checks the time arithmetic. The API response does
not supply each forecast's original run publication timestamp, so actual archive
vintage, changes in upstream models, and availability at cutoff have **not**
been individually verified. This is a historical forecast-input proxy, not a
fully gate-closure-audited operational backtest.

Brent is [EIA's Europe Brent spot series via FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU),
USD/barrel. The latest observation dated no later than 14 calendar days before
delivery and its seven-day change are used. This conservative delay does not
establish when each historical FRED value was actually published or whether it
was subsequently revised. FRED identifies the series as copyrighted; the cached
oil input, individual predictions and fitted models are local-only / Git-ignored.
No exchange rate or European gas price is included.

## Design and 2025 results

Two fixed model families: regularized logistic regression and histogram gradient
boosting. For each, compare calendar; calendar plus price history; price history
plus weather; and weather plus Brent. All ablations fit the same complete rows
and use equal total training weight for each pair. A pair-specific training rate
provides a simple baseline. No hyperparameter tuning uses 2025. From 2024 mean
pair Brier, the selected any-negative specification is gradient boosting with
weather and oil; the selected discordance specification is logistic regression
with price history. A 2024 selection followed by a 2025 check is exploratory
because both periods were previously inspected.

2025 any-negative Brier scores (lower is better):

| Pair | Baseline rate | Boost: price history | Boost: plus weather | Boost: weather and oil (2024-selected) |
| --- | ---: | ---: | ---: | ---: |
| DK1–DE-LU | .0639 | .0504 | .0487 | .0536 |
| DK1–NL | .0665 | .0523 | .0463 | .0529 |
| DK2–SE4 | .0435 | .0389 | .0372 | .0393 |

On the equal-pair mean, the 2024-selected weather+oil model changes 2025 Brier
from .0580 for the rate baseline to .0486. The weather-only version scores
.0441, but it was **not** the 2024-selected version; describing it as the winner
after seeing 2025 would be test-period selection. AUC and average precision are
reported with prevalence in `metrics.csv`. These are not directly comparable to
the earlier conditional-on-negative Brier scores because the population and
target changed.

Weather improves the boosted price-history Brier point estimate in all three
2025 pairs. In the 1,000-replicate seven-day paired-block intervals, DK1–NL's
gain is .0060 [ .0014, .0122 ]; the other two weather intervals cross zero.
Adding Brent to boosted weather worsens 2025 Brier for all three pairs, with
intervals below zero for DE-LU and NL. In the logistic family the oil ablation
instead improves the point estimates. These are exploratory comparisons with
multiple testing and selection, not a causal estimate or proof that oil never
helps. No interval corrects for model or hypothesis search.

The selected classifier's mean predicted any-negative probability is .018 in
DK1–DE-LU versus an observed .067 in 2025, and .019 versus .069 in DK1–NL;
probabilities are poorly calibrated under the year shift. For `discordant`,
the 2024-selected model's 2025 mean Brier (.01753) is only slightly better than
the pair-rate baseline (.01774). Rare-event occurrence and calibration therefore
remain unresolved even when the broader any-negative Brier improves.

## Outputs and reproduction

`metrics.csv` contains both outcomes and all families/features by pair/year,
including event counts, prevalence, average precision and ROC-AUC.
`ablation_intervals.csv` compares same-family, paired Brier losses with 1,000
seven-day blocks, separately for each pair/year. `input_coverage.csv` records
the complete-case population; `protocol.json` fixes the features and 2024
ranking. The forecast weather source is point data with its exact request URL,
retrieval timestamp and response hash stored in each gzip envelope.

Run `scripts/fetch_forecast_inputs.py` with network access, then
`scripts/train_forecast_models.py` from the repository root using the local
Python environment. FRED and the SE4 underlying paired data must be fetched
locally; see `data/multiarea/README.md`. The numerical experiments are research
prototypes and do not establish deployment-ready prices or publication-time
availability. Reloading the saved estimators reproduced all 16 fitted model
predictions for all 26,280 test pair-hours within 1e-9; the full test suite
passed (36 tests). Original underlying source attribution: Open-Meteo / JMA GSM;
EIA via FRED; Energinet; Energy-Charts / Bundesnetzagentur; eSett.
