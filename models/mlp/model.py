"""Multi-output feed-forward neural network baseline."""
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(80, 40), alpha=1.0,
                         max_iter=200, random_state=42, early_stopping=False))
