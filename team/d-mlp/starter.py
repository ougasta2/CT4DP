"""Reproduce the unchanged starter benchmark from a fresh checkout.

``model.py`` holds the tuned network, so ``scripts/run_team_benchmark.py
--model mlp`` no longer runs the starter, and ``--output`` only changes where
the tuned model writes. This script builds the starter configuration
explicitly -- two hidden layers (80, 40), L2 ``alpha=1.0``, a 400-iteration
budget, fixed seed, standardised inputs and targets -- and calls the same
shared runner, so ``results/team/mlp-starter/`` can be regenerated without
editing shared code::

    python team/d-mlp/starter.py

Expected scores (2023-2024, monthly refits): 2023 MAE 22.457 / RMSE 30.581,
2024 MAE 20.373 / RMSE 31.429.
"""

import sys
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_team_benchmark import OUT, SNAPSHOT, load_snapshot  # noqa: E402
from team_benchmark_core import run_prepared  # noqa: E402


def build_starter():
    """The starter exactly as shipped with the workspace."""
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(80, 40), alpha=1.0, max_iter=400,
                     random_state=42, early_stopping=False),
    )
    return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())


def main():
    features, targets, hourly = load_snapshot(SNAPSHOT)
    run_prepared('mlp-starter', build_starter, '2023-01-01', '2025-01-01',
                 features, targets, hourly, OUT / 'mlp-starter')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
