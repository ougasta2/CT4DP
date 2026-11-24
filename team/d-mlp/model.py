"""Multi-output neural network for the 24 local delivery hours.

The starting network was ``(80, 40)`` units with ``alpha=1.0`` and a
400-iteration budget. Rolling validation over the 2023 months
(``team/d-mlp/tuning.py``) selected one hidden layer of 40 units, ``alpha=10``
and an 800-iteration budget: the optimiser stops on its tolerance after about
268 epochs on average, so the larger budget never binds and produces the same
predictions as 400 in every refit.

Scaling of the 604 inputs and of the 24 targets is fitted inside the training
window only, so the benchmark's monthly refit never sees evaluation days.
"""

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """Build a fresh estimator that predicts prices in EUR/MWh."""
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(40,),
            alpha=10.0,
            max_iter=800,
            random_state=42,
            early_stopping=False,
        ),
    )
    return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())
