# Team workspaces in CT4DP

Work in this repository and open a pull request to `main` from your assigned
branch. Each member folder holds its model and any analysis specific to it.

| Member | Branch | Working folder | Responsibility |
| --- | --- | --- | --- |
| A | `team/a-lasso` | `team/a-lasso/` | Lasso |
| B | `team/b-random-forest` | `team/b-random-forest/` | Random Forest and clustering |
| C | `team/c-gradient-boosting` | `team/c-gradient-boosting/` | Gradient boosting and Apriori |
| D | `team/d-mlp` | `team/d-mlp/` | MLP |

From the repository root, follow the [shared benchmark instructions](BENCHMARK.md).
All models use the same snapshot and evaluator. Outputs go to the ignored
`results/team/<model>/` directory.

Member branches should change only their own folder. Changes to shared code,
data, protocol, automation, or other members' folders require a separate
lead-coordinated pull request. The `Team branch scope` workflow checks member
pull requests, including renamed paths. GitHub requires this check on `main`.
`ougasta2` decides when to merge each PR. The old repositories are archived,
and their commit histories are preserved on the `history/legacy-*` branches.
