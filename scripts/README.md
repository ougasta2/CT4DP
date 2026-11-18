# Script index

Run scripts from the repository root. The common team entry points are `run_team_benchmark.py` and `ensemble_team.py`; see the [root README](../README.md) for commands and the [team protocol](../docs/TEAM_WORKFLOW.md) for interpretation.

Other scripts implement earlier separate studies and retain their original names and paths. Broadly: `fetch_*` retrieves sources, `train_*` runs model experiments, `evaluate_*` analyzes saved predictions, `run_pilot.py` / `run_decade.py` / `run_followup.py` run historical analyses, and `verify_*` checks selected saved artifacts. These scripts can require local-only inputs, overwrite their own result folders, or use different forecast cutoffs. Read their source and corresponding result protocol before running.
