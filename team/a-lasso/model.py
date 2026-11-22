"""Sparse linear model for 24-hour DK1 price forecasts.

alpha=0.3 was chosen by the lowest rolling validation RMSE (2023 months) in lasso.ipynb.
"""

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """Return a fresh estimator with scaling fitted inside each training window."""
    return make_pipeline(StandardScaler(), Lasso(alpha=0.3, max_iter=10000))
