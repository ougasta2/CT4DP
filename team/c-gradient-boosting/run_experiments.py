"""Run single-change gradient-boosting variants through the shared benchmark.

Each variant changes exactly one setting of the shipped starter model. The
evaluator, inputs, dates and metric definitions are those of
scripts/run_team_benchmark.py. Choose a variant on 2023 only; report 2024.

    python team/c-gradient-boosting/run_experiments.py smaller_leaves
    python team/c-gradient-boosting/run_experiments.py --list
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from run_team_benchmark import SNAPSHOT, load_snapshot
from team_benchmark_core import run_prepared

BASE = dict(
    max_iter=60,
    learning_rate=0.08,
    max_leaf_nodes=15,
    max_bins=63,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42,
)

# One documented hypothesis per variant.
VARIANTS = {
    # Fewer leaves -> simpler trees, less overfitting to rare spikes.
    "smaller_leaves": {"max_leaf_nodes": 7},
    # Stronger L2 on leaf values -> shrinks extreme leaf predictions.
    "strong_l2": {"l2_regularization": 10.0},
    # Train on absolute error -> optimises the reported MAE directly and is
    # less pulled by price spikes than squared error.
    "absolute_loss": {"loss": "absolute_error"},
}


def make_builder(changes):
    params = {**BASE, **changes}
    return lambda: MultiOutputRegressor(HistGradientBoostingRegressor(**params), n_jobs=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=sorted(VARIANTS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()
    if args.list or not args.variant:
        for name, change in VARIANTS.items():
            print(f"{name}: {change}")
        return
    inputs = load_snapshot(SNAPSHOT)
    output = ROOT / "results" / "team" / f"gradient_boosting_{args.variant}"
    with threadpool_limits(limits=2):
        run_prepared("gradient_boosting", make_builder(VARIANTS[args.variant]),
                     args.start, args.end, *inputs, output, 365)


if __name__ == "__main__":
    main()
