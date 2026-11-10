# DK1 electricity-price forecasting: literature review and implementation plan

Reviewed 2026-11-10. This document recommends the next experiments; it does not
claim that any proposed model has been trained or has achieved the target.

## 1. Current problem and evidence

Predict DK1 day-ahead hourly average prices (EUR/MWh), using information available
by 10:00 Copenhagen time on the previous local day. The current strongest
explored candidate combines previous-day/week five-area price curves with
representative city temperature, wind and cloud forecast curves, using a
multi-output Ridge model. On 2025-11-10–2026-11-09 it attains 24.05% within
±5% (1,912/7,949 hours with |actual|≥20 EUR/MWh), and MAE 20.21 EUR/MWh over
all 8,760 hours. The same-architecture price-only comparison gives 21.73% and
24.16 EUR/MWh. This year was previously inspected and is an exploratory test;
the archived weather's historic publication/revision times are unverified.
[Existing matched experiment](../results/dk1_weather_ablation/findings.md).

A new descriptive cut of the same exploratory predictions is saved to
[segment_diagnostics.csv](../results/dk1_weather_ablation/segment_diagnostics.csv).
MAE is 11.69 EUR/MWh at 00–05 local clock hours (2,190 observations), 21.60
at 06–11, 24.85 at 12–17, and 22.69 at 18–23. Prices ≥200 EUR/MWh have MAE
40.99 (736 hours). Hours with actual prices <0 have MAE 25.96 (202 hours),
but only 15 qualify for the ±5% metric; the metric therefore barely measures
negative-price performance. These slices suggest where to investigate, but
they do not establish the causes of those errors.

## 2. What the primary literature supports—and what it does not

1. **Strong statistical benchmark and adaptation.** Lago et al. (2021),
   [Forecasting day-ahead electricity prices](https://doi.org/10.1016/j.apenergy.2021.116983),
   develops open LEAR (LASSO-estimated autoregression) and DNN benchmarks. Its
   LEAR uses the previous 1, 2, 3 and 7 daily price curves, predictors across
   the whole day, an asinh transformation, and daily recalibration; its ensemble
   averages 8-week, 12-week, 3-year and 4-year calibration windows. The
   benchmark includes Nord Pool, but its original test data predate the recent
   energy crisis and do not show DK1 ±5% performance. This is a good first
   comparator to our fixed Ridge, not proof it will win in the current market.
2. **The connected system matters.** Van der Heijden et al. (2021),
   [market integration](https://doi.org/10.1109/ACCESS.2021.3108629), report
   that selected European-market features improved their strongest Dutch model's
   MAE by 3.1% relative to their no-European-features reference. This is a
   Dutch case, not a DK1 estimate. We already include five regions' lagged
   prices; the incremental question is whether legally usable pre-auction
   forecasts, congestion/capacity and price spreads add value beyond them.
3. **Uncertainty in renewable and load forecasts can matter.** The paper
   [Probabilistic Forecasts of Load, Solar and Wind for Electricity Price
   Forecasting](https://arxiv.org/abs/2501.06180) reports improved German
   day-ahead point forecasts when probabilistic load and renewable forecasts
   are incorporated. It is evidence for testing forecast spread and nonlinear
   interactions, not evidence of DK1 gains or that point forecasts suffice for
   correctly calibrated predictive distributions. In our case reliable
   historic pre-cutoff vintages of these inputs still need to be established.
4. **Nonlinear models are candidates, not automatic winners.** The 2026
   [cross-border forecasting preprint](https://arxiv.org/abs/2608.17091)
   finds gains from neural models over simple baselines, but those baseline
   models do not include the full LEAR benchmark; the study also reports the
   best among several training-window configurations for each architecture.
   Its numbers are not comparable to ours. Compare a small feed-forward DNN
   using precisely the same data and temporal evaluation as LEAR and Ridge
   before considering larger sequence models.
5. **Predicting the full price distribution is useful for a ±5% decision.**
   [Learning the probability distributions of day-ahead electricity prices](https://doi.org/10.1016/j.eneco.2025.108988)
   and the probabilistic benchmark by
   [Marchesi et al.](https://www.mdpi.com/1996-1073/18/23/6269) motivate
   distributional forecasts and calibration. They do not demonstrate that a
   mode or interval-optimal point estimate automatically increases ±5% hits.
   Calibrated distributions, proper scores and the actual hit metric must all
   be evaluated; a more concentrated forecast alone can be badly calibrated.
6. **Supply/demand curves are promising but a longer-term experiment.** The
   [merit-order curve paper](https://arxiv.org/abs/2512.17758) predicts
   historical auction curves and derives clearing prices. Its 2024 test shows
   gains in Italy for some curve models, but price-based LEAR and fARX
   outperform its curve-based alternatives in DE-LU; fit of curves also does
   not necessarily imply better price estimates. This is a preprint and its
   results do not establish DK1 performance. Nord Pool's
   [aggregated curves](https://data.nordpoolgroup.com/auction/day-ahead/aggregated-bid-curves?market=Day-ahead)
   include realized block volumes and net flows. Same-auction results are
   unavailable before that auction: use only historical curves known by the
   cutoff and audit their timestamps, access and licensing before attempting a
   structural model.

## 3. Recommended design and priority

**Phase A: make comparisons trustworthy.** Keep the current matched Ridge and
previous-day/previous-week baseline. Use one dataset keyed by delivery UTC,
local day, cutoff and source/vintage; retain raw observation and availability
or first-seen timestamps. Add contract tests that reject information released
after cutoff and missing hours. Unverified weather is a clearly labeled
exploratory tier, not equivalent to an audited live backtest. Past day-ahead
auction results are assumed to have appeared normally before the cutoff;
exceptional delays and corrections remain to audit. At a strict 10:00 cutoff,
a capacity announced at 10:00 is usable only if publication and ingestion are
known to precede the actual forecast timestamp; otherwise use an earlier
snapshot or move the forecast time and rerun all models consistently.

**Phase B: high-priority model experiment with current data.** For Ridge and
LEAR, add price curves from d−2 and d−3 to existing d−1 and d−7, past price
spreads and daily amplitude, with calendar and existing forecast weather.
LEAR estimates sparse per-hour regressions; compare a fixed-window and a
rolling/recalibrated version with windows 8 weeks, 12 weeks, 3 years and 4
years and their equal-weight ensemble. Tune only inside prior rolling
validation, and compute transformations/scalers from each training window.
Evaluate optional asinh price scaling as an ablation: it handles negative
prices but the inverse transform and squared-error objective can change bias.
Short windows are data-starved; keep strong regularization and compare them
against the no-update Ridge, updating only with outcomes known by each cutoff.
Because different windows alter training data, separately report the gain from
extra features and from rolling recalibration; keep dates and inputs matched
within each direct comparison. Before spending on complex neural models,
train a modest multi-output feed-forward network and ensemble seeds/models with
weights chosen solely on out-of-fold historical predictions. Neither an
ensemble nor a rolling window is assumed to improve performance.

**Phase C: information that explains daytime and ramps.** Add audited
pre-cutoff forecast curves for DK1/DK2/DE-LU (expand coverage only where
available): load, onshore/offshore wind, solar, and residual load = predicted
load − predicted wind − predicted solar. Include their day-over-day change,
forecast spread if a genuine pre-cutoff ensemble is available, and before/after
hour ramps. Obtain archived cross-border available capacity/outages, prior-day
spreads and dated fuel/EUA quotations where point-in-time access is verifiable;
all features must match their target zones and physical units. Prioritize TTF
and EUA over Brent as a testable marginal-cost hypothesis, not a claim that oil
is irrelevant or gas always sets DK1 prices. The existing lagged Brent/gas
proxies have unverified release/revision times and are exploratory. Never use
realized generation, the same day's realized flow or the same-auction clearing
price as ex ante inputs. Run matched ablations: prices; +weather; +audited
load/renewables; +audited fuels/capacity. Evaluate MAE, ±5% and subgroup
performance at each step, not just feature importance. The relative timing,
revision history, coverage, access and provenance of each candidate must be
reported; a day-ahead label alone is insufficient.

**Phase D: forecast distribution and decision rule.** Produce quantiles or a
conditional density using the same audited information and time-split
training; calibrate with rolling out-of-fold predictions. Compare MAE-oriented
median and a point chosen to maximize estimated probability of
`|prediction−Y| ≤ 0.05|Y|` for `|Y|≥20`. Calculate that probability from the
fitted distribution, not by plugging a prediction into the unknown denominator.
Use chronological out-of-fold calibration for any residual model or ensemble;
measure CRPS/pinball loss and empirical coverage/width in addition to point
scores. Select an improvement in development only if MAE and extreme/negative
price errors are acceptable as well; optimizing a narrow hit band can sacrifice
other hours. Time dependence and regime shifts mean conventional conformal
coverage cannot be assumed without checking its assumptions.

**Later, not first:** reconstruct past bid curves and constrained auction
fundamentals if acceptable coverage, licensing and point-in-time histories are
available. Test against the same LEAR/DNN models. Explore a separate 15-minute
product after sourcing consistent historical auction intervals; never pretend
that four identical hourly predictions are 15-minute accuracy. Document the
market transition, DST 92/96/100-slot days and any shortened usable history.

## 4. Evaluation and acceptance gates

- Use expanding or rolling walk-forward development with folds spanning quiet
  and stressed periods (including 2022, 2023 and 2024–2025), keeping all
  preprocessing, feature selection, calibration and model choice within the
  earlier training portion. Report each fold and same-sample matched baselines;
  no random cross-validation. Any labels included for a forecast must have
  actually been published before its cutoff.
- The previously analyzed 2025-11-10–2026-11-09 year is exploratory only. Freeze
  the experiment and reserve subsequent, presently unseen delivery days as a
  prospective test. Report preliminary results as data accumulates; do not
  call a full-year evaluation complete until one year has actually elapsed.
- Primary outcome: share with |prediction−actual|≤0.05|actual| among hours
  with |actual|≥20 EUR/MWh, with eligible counts. Guardrails: all-hour MAE,
  price-band/hour-of-day breakdowns, negative-price errors and high-price
  errors; probability models additionally require CRPS and coverage/width.
- Pre-register paired comparisons, count all model searches, and report
  uncertainty from paired day/block bootstrap or another dependence-aware
  method; selection and repeated analyses limit inference. Prefer a model only
  if it improves the primary metric consistently across development folds
  without material deterioration in the guardrails and then retains the gain
  on the future holdout. This is a decision rule, not a guarantee that the
  uncertainty interval will establish a significant improvement.
- No source reviewed here establishes that DK1 can achieve a particular ±5%
  success rate. If a sufficiently large prospective improvement is absent,
  report that and revisit the data and decision horizon rather than selecting
  the most favorable previously observed result.
