# Script index

Run scripts from the repository root. The common team entry points are `run_team_benchmark.py` and `ensemble_team.py`; see the [root README](../README.md) for commands and the [team protocol](../docs/TEAM_WORKFLOW.md) for interpretation.

`postprocess_team.py` generates the full-period model comparison, blends, and
frozen Apriori rule tables. Run `python scripts/postprocess_team.py --help` for
the five prediction-file arguments and the regime-label argument. Supply the
2023–2024 predictions on the shared hourly snapshot and B's daily K-means
labels. The default output directory is `results/team/postprocess/`.

The [report appendix notebook](../notebooks/team_postprocess.ipynb) uses the
frozen forecast and sensitivity files bundled in the submission appendix under
`team/report_inputs/`. Run it from the extracted appendix using the pinned
dependencies and the instructions in its README.

Other scripts implement earlier separate studies and retain their original names and paths. Broadly: `fetch_*` retrieves sources, `train_*` runs model experiments, `evaluate_*` analyzes saved predictions, `run_pilot.py` / `run_decade.py` / `run_followup.py` run historical analyses, and `verify_*` checks selected saved artifacts. These scripts can require local-only inputs, overwrite their own result folders, or use different forecast cutoffs. Read their source and corresponding result protocol before running.
