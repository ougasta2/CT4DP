"""Multi-output ridge regression baseline."""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
