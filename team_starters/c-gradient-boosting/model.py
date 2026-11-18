"""One histogram gradient boosting regressor per local delivery hour."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_model():
    """Return an unfitted estimator for the common daily benchmark."""
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=60,
            learning_rate=0.08,
            max_leaf_nodes=15,
            max_bins=63,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        ),
        n_jobs=1,
    )
