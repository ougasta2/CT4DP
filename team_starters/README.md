# Archived starter templates

The active member workspaces are in [team/](../team/README.md) within this
repository. Do not create new standalone repositories from these templates.
The instructions below describe the former standalone setup for reference only.

The lead maintains the frozen Ridge baseline and integrates the ensemble.
A owns Lasso, B owns Random Forest and Week 6 clustering, C owns histogram
gradient boosting and Week 5 Apriori, and D owns the MLP. Each member starts
with an independent Git repository, an English assignment and the same
benchmark. The model benchmarks can run independently. C can start its
calendar-only rule analysis immediately and compare it with cluster-containing
rules once B supplies the market-regime labels.

To assemble the four local repositories from the parent project, first set up
the data described in the [catalog](../docs/DATA_CATALOG.md), then run:

```powershell
.venv/Scripts/python scripts/run_team_benchmark.py --model ridge
.venv/Scripts/python scripts/prepare_team_repos.py
```

The second command creates `team_repos/` and initializes one Git repository
per member. It refuses to overwrite that directory. Use `--destination` with
a new path to generate another set. The main project tracks the starter
sources and ignores the generated repositories. Each generated repository
works without the parent checkout and can track the four allowlisted snapshot
files. See [data sources and licence](DATA_LICENSE.md). Keep results out of Git.
