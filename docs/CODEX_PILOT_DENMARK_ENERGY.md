# Codex task: Pilot study on Danish renewable generation and electricity prices

## Your role and objective

Act as a careful Python data scientist working in the current repository. **Implement and actually run a reproducible, appropriately scoped pilot** for a DTU course project (02807 Computational Tools for Data Science). Do not stop after outlining a plan or generating empty notebooks. Inspect the repository first, reuse any relevant existing code, then implement, execute, test, and report what the data supports.

**Study question:** *Which combinations of wind generation, solar generation, and electricity demand are associated with negative or unusually high day-ahead electricity prices in Denmark, and do the discovered patterns transfer from 2023 to 2024?*

Compare the Danish electricity price areas **DK1 and DK2**. This is an **exploratory association analysis**, not a causal study or a deployable day-ahead forecasting model.

**Deliverable language:** code and technical documentation in English. Prefer concise, interpretable plots. No web dashboard, cloud infrastructure, or heavy ML/deep-learning stack.

## Course constraints (must shape the work)

The 02807 final project requires:

- At least one, preferably **two**, substantial topics from the course. Use **K-means clustering** and **frequent itemsets / Apriori association-rule mining** as essential parts of this pilot. Merely calling a library with little substantive analysis is insufficient.
- At least **one clearly identified method not taught in the lectures**. For this pilot, implement a **time-series block bootstrap** for uncertainty/robustness assessment of selected rules (if feasible), or document a technically justified replacement. Keep it separate from the course topics.
- Some nontrivial independent implementation. **Implement the Apriori candidate-generation/counting core yourself** on the small discretized dataset, and verify it against hand-worked unit tests or a trusted reference for a small sample. Use `scikit-learn` for K-means rather than reimplementing it.
- A rigorous evaluation, discussion of limitations, and runnable, documented Jupyter Notebook. The eventual report is concise (4-person group: maximum 11,000 characters excluding whitespace; 5-person group: 12,500). **Do not write the full final report now.**

The assignment PDF has conflicting hand-in language (December 4 vs. December 6, 2026). Plan against the earlier date, and note that the team should check DTU Learn.

## Data: use verified official sources

Use **Energinet Energi Data Service**. Discover and validate schema from live metadata before fetching large amounts; do not silently invent or substitute fields.

1. **Production and consumption settlement**:
   - API: `https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement`
   - Metadata: `https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement`
   - Key fields: `HourUTC`, `PriceArea`.
   - Wind components (MWh, per one-hour interval):
     - `OffshoreWindLt100MW_MWh`
     - `OffshoreWindGe100MW_MWh`
     - `OnshoreWindLt50kW_MWh`
     - `OnshoreWindGe50kW_MWh`
   - Solar components (MWh):
     - `SolarPowerLt10kW_MWh`
     - `SolarPowerGe10Lt40kW_MWh`
     - `SolarPowerGe40kW_MWh`
     - `SolarPowerSelfConMWh` (estimated self-consumption; analyze/flag separately)
   - Demand proxy: `GrossConsumptionMWh` (MWh/hour). Consult source metadata for its definition and uncertainty.
2. **Historical day-ahead electricity prices**:
   - API: `https://api.energidataservice.dk/dataset/Elspotprices`
   - Metadata: `https://api.energidataservice.dk/meta/dataset/Elspotprices`
   - Key fields: `HourUTC`, `PriceArea`; price: `SpotPriceEUR` (EUR/MWh).
   - This is a **discontinued dataset after September 30, 2025**; this is acceptable because the pilot only uses 2023 and 2024. Do **not** quietly combine it with the post-2025 15-minute dataset.
3. Official API guide: `https://www.energidataservice.dk/guides/api-guides`
   - Supports `start`, `end` (exclusive), `filter`, `columns`, `limit`, `offset`, `sort`.
   - Typical filter: `filter={"PriceArea":["DK1","DK2"]}` as JSON-encoded query argument.
   - Pagination response contains `records` and usually `total`.
   - API's `HourUTC` strings **have no trailing Z even though they represent UTC**. Explicitly parse as UTC, then convert to `Europe/Copenhagen` only when deriving local calendar features. Merge on `(HourUTC, PriceArea)`, not `HourDK`, to avoid daylight-saving ambiguity.
   - Follow rate-limit guidance, cache responses locally, and implement bounded retries for 429 / transient failures.

**Important scientific details:** Use the physically correct unit labels (MWh per hourly interval, EUR/MWh). `GrossConsumptionMWh` is not a perfectly independently measured demand series. Solar self-consumption is estimated. Distinguish solar generation delivered to the grid from total solar including estimated self-consumption. Never replace missing observations with zero without an explicit, defensible rule. Do not conflate actual settlement generation with information available before the day-ahead market clears.

## Scope: minimal working pilot first

- **Quick smoke test:** request a single seven-day window from 2024 for DK1 and DK2, inspect returned columns and sample values, and demonstrate that prices and production can be joined correctly.
- **Main pilot:** use all of **calendar year 2023 for training** and **calendar year 2024 for out-of-time validation**, both DK1 and DK2. Decide year membership from local Copenhagen timestamps; take care of year-boundary UTC intervals. If downloads fail or completeness is inadequate, fall back to a smaller documented period but do not imply a full-year study was completed.
- No 2025+ data, no external weather APIs, no installed-capacity data, no price forecasting, no power-grid optimization at this stage.

### Suggested pipeline

1. **Acquire and validate.** Fetch and locally cache both datasets by bounded intervals (e.g. by month). Use a stable ordering or time-window partitioning to avoid pagination omissions. Record request parameters, source URLs, download dates, counts, and schema. Fail visibly on unexpected schema.
2. **Clean and join.** Aggregate `wind_mwh = sum(four wind components)` and `solar_grid_mwh = sum(three solar-grid components)`. Optionally compute `solar_total_est_mwh = solar_grid_mwh + SolarPowerSelfConMWh` as a *sensitivity variant*, not as a silently definitive measurement. Use `gross_load_mwh = GrossConsumptionMWh`. Verify one row per `(HourUTC, PriceArea)` per dataset, then inner-join with a merge audit (record unmatched and missing percentages; do not silently drop them). Set an explicit policy for incomplete rows. Sort chronologically and retain only usable values.
3. **Engineer features.** For each area-hour compute `wind_to_load = wind_mwh / gross_load_mwh` and `solar_to_load = solar_grid_mwh / gross_load_mwh` for strictly positive load, **without clipping ratios at 1**. Build `load_index = gross_load_mwh / training_median_load_for_that_area` (fit medians on 2023 only). Optionally include onshore/offshore composition. For interpretation and baselines derive `local_hour`, `month`, `weekday`, `season`, and a broad `time_of_day` bin from Copenhagen local time. **Never include price or price-derived variables in clustering features.**
4. **K-means (course topic 1).** Standardize only with 2023 training data, train a common K-means on DK1+DK2 using the three primary numeric generation/load features. Compare `k` in 3–7 with Davies–Bouldin and sampled silhouette scores and examine cluster sizes/stability. Do not mechanically choose a pathological tiny-cluster solution. Freeze scaler/model and assign 2024 clusters. Describe cluster centers in original physical features and give descriptive names only *after* inspecting results.
5. **Association patterns using Apriori (course topic 2).** Convert each hourly record into a small item basket: `cluster=C#`, `area=DK1/DK2`, training-quantile-based `demand=low/mid/high`, coarse `time_of_day`, and `season` (optionally solar/wind bins if redundancy is explained). Each categorical dimension contributes at most one item. Mine **predictor-only itemsets** with custom Apriori (max size 3; configurable minimum antecedent support such as 1–2%; count exact support). After mining, independently score candidate rules `antecedent => price_event` so rare outcome events are **not accidentally pruned** by the frequent-itemset threshold. Rank rules by support, confidence, lift, and sample count. Limit the final shortlist to a few interpretable rules, not thousands. Use the K-means cluster items in the candidate-generation pipeline so both course methods contribute substantively.
6. **Price events and holdout.** Define `negative_price = SpotPriceEUR < 0`; define `high_price = SpotPriceEUR > training-2023 95th percentile within each area`. Fix the 2023 thresholds, feature bin edges, trained cluster labels and selected rules before looking at 2024 performance. Report event counts/base rates by year and area. Score fixed rules on 2024 using support, event rate, lift relative to the 2024 area-appropriate base rate (and differences from training). Compare a simple calendar-only grouping baseline, such as `area + season + time_of_day`, to check if energy-regime information adds insight beyond seasonality. Distinguish **association discovery** from predictive forecasting.
7. **Uncertainty and failure modes.** For the top few selected rules, assess how much their 2024 event-rate lift varies using a **moving/block bootstrap of consecutive days or weeks** (keeping neighboring hours together, not i.i.d. hourly resampling). Clearly label exploratory uncertainty intervals, and explain limits (serial dependence, overlapping items, multiple comparisons, and rare event counts). If the selected rules do not generalize or negative-price events are too rare, **report that result**, not invented success. Check how conclusions change if estimated solar self-consumption is included.

### Explicit analysis checks

- Data coverage by month, year, price area, source dataset; duplicate and missing timestamps; join retention rate.
- DST and boundary checks. 2024 is a leap year. Do not assume every Copenhagen local day has exactly 24 hours.
- MWh sums are nonnegative where physically appropriate; verify missing versus legitimate zeros. Identify outliers but do not automatically remove negative prices.
- Event threshold leakage and model fitting leakage: no transformations or thresholds learned on 2024.
- K-means cluster size and separability, rule support counts, event-rate denominators, and minimum sample-size flags.
- Failure states should produce descriptive errors and diagnostics rather than silently returning empty plots or plausible-but-false numbers.

## Engineering deliverables

Create or update the following without overwriting unrelated user files. Adapt structure if the repository already has a convention:

```text
README.md                          # how to install, run, reproduce and limitations
requirements.txt OR pyproject.toml # light Python dependencies, pinned or constrained
src/energy_pilot/                 # reusable API, preprocessing, features, mining, evaluation modules
scripts/run_pilot.py              # one reproducible CLI entry point
notebooks/energy_price_pilot.ipynb # concise, runnable figures and explanations
results/figures/                  # charts produced from REAL downloaded data
results/pilot_findings.md          # results with numerators/denominators, caveats, decision for next iteration
tests/                             # unit tests, especially Apriori, timestamps and join logic
```

Suggested CLI: `python scripts/run_pilot.py --smoke-test` and `python scripts/run_pilot.py --train-year 2023 --test-year 2024`. Provide `--offline` for cached-data reruns and `--max-records`/`--small-sample` if useful for debugging. Use `requests`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pytest`, optionally `scipy` and `pyarrow`. No credentials should be needed.

Minimum plots from real data:
- Renewable generation and day-ahead price over one representative week (both areas, clearly labeled axes).
- Scatter/hexbin of wind-to-load vs solar-to-load with price event rate, or a suitable equivalent.
- Interpretable clustering result (centroid feature profile and cluster sizes, with area breakdown).
- 2023 vs 2024 event rates/lifts for the short list of rules, with base-rate references and uncertainty when possible.

## Acceptance criteria — do the work, do not just promise it

1. `--smoke-test` **executes** and prints the source schemas, records fetched, join statistics, and at least a few actual merged observations, or reports precisely which network/API step blocked execution.
2. The full pilot runs from cached or live data without arbitrary manual file edits, prints 2023/2024 record counts, trains K-means, mines and evaluates rules, and writes at least two valid real-data figures.
3. Unit tests cover Apriori counts/candidate pruning, rare-outcome scoring, UTC/DST handling, and duplicate-key/merge assertions; run the tests and record results.
4. `results/pilot_findings.md` answers: **What did we discover? Does it generalize to 2024? Does it outperform the calendar-only baseline? Are negative-price findings adequately supported?** Include real numbers only after actual execution.
5. `README.md` describes how the official datasets were acquired, appropriate citations and data license attribution (Energinet, CC BY 4.0), reproducibility, and the limits of historical associations.
6. If live API/network access is unavailable, **complete offline-testable code and tests**, mark all empirical results as **not run/not verified**, and provide the exact command for later execution. Never invent observations or present synthetic fixtures as empirical findings.

## Execution order / stop rule

- First inspect the current repository and the official metadata endpoints. State any blocking ambiguity in a short implementation note, then proceed with reasonable defaults.
- Run a tiny network smoke test **before** writing the whole pipeline. If it fails, focus on robust implementation and offline fixture tests.
- Deliver a **working vertical slice** (data -> clean join -> one K-means -> one Apriori rule -> one plot) before adding full years, bootstrap intervals, or cosmetic polish.
- Prioritize scientific correctness and demonstrable course-algorithm relevance over quantity of plots.
- At the end, summarize files changed, exact commands and observed execution/test results, preliminary scientific findings (if genuinely computed), and three or fewer prioritized next steps.

**Do not:** invent API results, claim causal effects, claim live price prediction from realized generation, overfit by repeatedly tuning rules on the 2024 test set, fill gaps with made-up data, or broaden the pilot into battery storage/optimization.

## Primary references

- Energinet dataset metadata: https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement
- Energinet price metadata: https://api.energidataservice.dk/meta/dataset/Elspotprices
- Official API guide: https://www.energidataservice.dk/guides/api-guides
- Energinet data use terms: https://www.energidataservice.dk/support-service
- Course assignment: DTU 02807 Final Project, Autumn 2026 (the user can provide `project.pdf` if needed).
