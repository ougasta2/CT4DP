# Execution record — October 31, 2026

All commands were run from the repository root with the local Python 3.10.11
virtual environment. Exact installed packages are in requirements-lock.txt.

## Observed executions

```powershell
.venv/Scripts/python scripts/run_pilot.py --smoke-test
.venv/Scripts/python scripts/run_pilot.py --train-year 2023 --test-year 2024
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/build_notebook.py
.venv/Scripts/python scripts/execute_notebook.py
```

- Smoke test completed: 336 production and 336 price rows, all 336 joined;
  no missing cells, duplicate keys or invalid physical cells. Real-data
  clustering, Apriori scoring and three diagnostic figures completed.
  Its temporary within-week split is not empirical year-transfer evidence.
- Full live pilot completed: 17,520 area-hours in 2023 and 17,568 in 2024.
  Both sources have complete monthly coverage in both areas, no null source
  cells and 100% join/cleaning retention. Four primary real-data figures written.
- 13 unit/integration tests passed (5.37 seconds in the recorded final suite).
  See test_results.txt. Tests include exhaustive Apriori comparison, rare-outcome
  scoring, duplicate/missing joins, UTC/DST/leap handling, train-only features,
  bootstrap reproducibility, calendar fallback, cache/retry/schema failures and
  a no-event analysis that must return no selected rules. Synthetic fixtures are
  used only in tests and temporary directories, never empirical findings.
- The 19-cell notebook executed top-to-bottom, with all 9 code cells executed,
  no error outputs and four embedded figures. It executes the main CLI with
  `--offline`, demonstrating cached-data reproducibility. It was refreshed after
  the findings tables were changed to Markdown for readability.
- Independently checked that the four selected rules match
  pre_holdout_selection.json, saved before the 2024 price download completed.
- Primary PNG files were opened and visually inspected for axes, units, labels,
  comparisons and uncertainty. PNG decoding verified. The week chart now uses
  shared scales within energy/price columns for truthful area comparisons.
- The notebook's saved code/output structure and embedded plots were inspected;
  its complete layout has not been opened in a Jupyter/VS Code notebook viewer.
  To complete that presentation check, open notebooks/energy_price_pilot.ipynb
  in either viewer and inspect the rendered Markdown tables and four figures.

## Execution limitations and recovery

The initial sandbox network attempt failed DNS. Authorized network-enabled
execution reached the official sources. The discontinued price API returned
HTTP 429; bounded retries honored the specified waits (including 277 and 299
seconds) and completed successfully. last_failure.json preserves an **earlier,
recovered** smoke failure, not the final status. run_manifest.json records the
latest successful cached execution.

The restricted process could not start a responsive local notebook kernel;
the authorized local Jupyter execution succeeded. No global Python packages
were installed. Network access is not needed for a fully cached rerun.

These are execution checks, not a causal validation. All four selected-rule
bootstrap intervals include lift 1. The negative-price shortlist does not
retain point-estimate enrichment in 2024; one primary rule has only 17 validation
events. Calendar+cluster improves negative-event Brier score but worsens the
high-price score. See pilot_findings.md for exact numerators and denominators.
