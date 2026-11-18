# Model workspaces

Each folder contains an independently editable `model.py` with a `build_model()` function. It must return an unfitted scikit-learn-compatible estimator accepting a daily feature matrix and a 24-column daily target matrix. The shared runner owns data preparation, chronological splits, evaluation, timing and output. Keep this interface when improving a model; put model-specific preprocessing inside the returned estimator so it is fitted on the training data only.

| Workspace | Starting point | Suggested investigation |
| --- | --- | --- |
| [ridge](ridge/) | Regularized linear regression | regularization, training window, transformations |
| [lasso](lasso/) | Sparse linear regression | regularization, stability, lag selection |
| [random_forest](random_forest/) | Nonlinear tree ensemble | tree size, regularization, feature interactions |
| [mlp](mlp/) | Feed-forward neural network | architecture, regularization, scaling |

These are starting implementations, not claims that every model has already been benchmarked under the new protocol. Do not call the Lasso implementation an exact LEAR reproduction. Compare all changes on the same dates; record failed experiments too. See [the experiment protocol](../docs/TEAM_WORKFLOW.md).
