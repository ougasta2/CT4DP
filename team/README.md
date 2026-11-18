# Team workspaces in CT4DP

These folders contain the tracked contents of the former individual member
repositories, including their benchmark snapshots and attribution. Work in
this repository and open a pull request to `main` from your assigned branch.

| Member | Branch | Working folder | Responsibility |
| --- | --- | --- | --- |
| A | `team/a-lasso` | `team/a-lasso/` | Lasso |
| B | `team/b-random-forest` | `team/b-random-forest/` | Random Forest and clustering |
| C | `team/c-gradient-boosting` | `team/c-gradient-boosting/` | Gradient boosting and Apriori |
| D | `team/d-mlp` | `team/d-mlp/` | MLP |

For each workspace, enter its directory, install `requirements.txt`, then run
`python benchmark.py --start 2023-01-01 --end 2023-02-01`. Its README describes
the full benchmark and its files. Outputs remain ignored by that folder's
`.gitignore`. The common research protocol is in
[docs/TEAM_WORKFLOW.md](../docs/TEAM_WORKFLOW.md).

Member branches should change only their own folder. Changes to shared code,
data, protocol, automation, or other members' folders require a separate
lead-coordinated pull request. The `Team branch scope` workflow checks member
pull requests, including renamed paths. For enforcement, a repository admin
must protect `main`, require this check and pull requests, and prevent bypasses.
GitHub branch names and this check alone do not authenticate folder ownership;
use GitHub permissions, branch rules, and review to manage who may push or merge.

The individual local copies and their Git histories are retained until their
remote state has been verified and migration is complete.
