# Extended-study execution record — November 1, 2026

## Observed runs

```powershell
.venv/Scripts/python scripts/run_decade.py --offline
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/build_decade_notebook.py
.venv/Scripts/python scripts/execute_notebook.py notebooks/energy_price_decade.ipynb
.venv/Scripts/python scripts/index_decade_data.py
```

- Downloaded and cached official 2016–2022 production/hourly-price bulk windows,
  2025 production, January–September 2025 hourly prices, and October–December
  quarter prices. Reused the original 2023/2024 cached years. Official schema was
  validated before fetching; the last discontinued-price request honored its
  295-second Retry-After. Inputs are bundled; later runs need no API access.
- Complete source time keys and required selected fields in both areas for
  all ten years. No unmatched join keys. Exactly 175,344 source area-hours per
  hourly dataset: 140,256 training rows (2016–2023), 35,088 test rows (2024–2025).
  Every expected monthly key and each year/area usable count matched.
- Identified 1,747 negative estimated self-use values in 2020 (876 DK1, 871 DK2).
  Applied the metadata's nonnegative estimate bound with original values and
  repair flags retained. Strict-no-repair and no-solar clustering sensitivities
  were run on the same fixed holdout targets/population. No missing values were
  imputed as zero. Original production exports retain signed values.
- 17,672 unique quarter prices form 4,418 complete hourly area observations for
  October–December 2025, including the 25-hour local DST day. Hourly means were
  negative in 45 observations; 51 had at least one negative quarter. Thus six
  observations differ under the hourly-mean versus any-quarter definitions.
- K-means baseline, GMM, DBSCAN, Random Forest classification and histogram
  gradient boosting regression all fitted and evaluated. Both supervised methods
  have same-family calendar-only ablations. No random temporal split or holdout
  tuning. Models, preprocessing and all 12 Apriori rules were frozen before
  held-out target scoring; see frozen_protocol.json and its manifest checksum.
- 26 tests passed (7.01 seconds in the observed suite). New tests cover complete
  quarter averaging, duplicate/missing quarters, DST, explicit solar bridge and
  negative-estimate repair, eight/two chronological splitting, frozen DBSCAN core
  extension/noise, metric sampling with only one observed label, no target/year
  leakage in supervised inputs, year/area sampling, training-only calendar
  backoff and equivalence of daily sufficient-count bootstrap to hourly weights.
- Reloaded the saved fitted models and reproduced **every one of the 35,088
  holdout** K-means/GMM/DBSCAN assignments, Random Forest probabilities and
  regression estimates within floating-point tolerance (rtol=1e-10). This check
  used the shared processed table, not an API requery.
- All 22 checksum entries (16 original cached responses, six derived artifacts)
  verified. Full source values and derived data are available for reproduction.
- Exported all 128 staged Git files to a fresh temporary directory. With HTTP
  requests explicitly disabled, the isolated copy reconstructed all 175,344 raw
  hourly rows, the chronological eight/two split and every test cluster label;
  its saved Random Forest reproduced all negative-price probabilities directly
  from harmonized raw cache inputs. No original-workspace ignored file was copied.
  All checksums and the frozen-protocol hash also matched the exact Git blobs.
  `.gitattributes` and canonical-LF JSON keep those hashes stable on Windows,
  Linux and GitHub ZIP downloads.
- The 21-cell notebook ran top-to-bottom: 10 code cells, five embedded PNG figures,
  no error outputs. It loads executed outputs by default; RERUN_ANALYSIS=True
  invokes full offline model refitting. Full fitting was separately executed via
  the CLI above. The original two-year notebook remains unchanged.
- Opened and visually inspected all five saved figures. Corrected clipped price-
  regression titles and shortened labels, regenerated from unchanged scored
  outputs and re-executed the notebook. All PNGs decoded. The complete notebook
  layout has not been opened in a Jupyter/VS Code viewer; open its ipynb to finish
  that presentation-only check.

## Results and interpretation checks

Random Forest negative-event Brier improves 0.036060 -> 0.032687 when energy
features are added; paired seven-day bootstrap improvement interval
[0.001885, 0.005136]. High-event Brier worsens 0.012078 -> 0.022764, and boosting
MAE worsens 35.65 -> 41.12 EUR/MWh. Negative K-means solar-rich afternoon rules
retain enrichment in both individual test years. 88.4% of training high-price
events occur in 2022, demonstrating a substantial distribution shift rather than
stable high-price rules. GMM and DBSCAN do not provide uniformly stronger cluster
separation; DBSCAN holdout noise is 45.8%. These failures are retained.

2024 was inspected in the earlier pilot. The test is exploratory out-of-time
evaluation, not a new pristine confirmatory sample. Realized settlement features
are not pre-clearing information; no operational forecasting or causal claim is
supported. Degenerate zero-event bootstrap intervals do not establish zero future
event probability. See pilot_findings.md for full denominators and limitations.
