"""Sparse linear baseline; LEAR-inspired, not a reproduction of LEAR."""
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    return make_pipeline(StandardScaler(), Lasso(alpha=0.03, max_iter=3000, random_state=42))
