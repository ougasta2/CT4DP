"""Multi-output neural network for the 24 local delivery hours."""

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """Build a fresh estimator that predicts prices in EUR/MWh."""
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(80, 40),
            alpha=1.0,
            max_iter=400,
            random_state=42,
            early_stopping=False,
        ),
    )
    return TransformedTargetRegressor(
        regressor=network, transformer=StandardScaler()
    )
