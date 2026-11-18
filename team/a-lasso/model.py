"""Sparse linear starter for 24-hour DK1 price forecasts."""

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """Return a fresh estimator with scaling fitted inside each training window."""
    return make_pipeline(StandardScaler(), Lasso(alpha=0.03, max_iter=3000))
