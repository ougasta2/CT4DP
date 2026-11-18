"""Nonlinear multi-output random forest baseline."""
from sklearn.ensemble import RandomForestRegressor

def build_model():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=42)
