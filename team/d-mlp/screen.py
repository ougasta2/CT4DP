"""Stage-1 screen: single chronological split (fit 2019-2022, score 2023).

A 12-month rolling validation fit per setting is the protocol used for the
final choice, but it is too slow to explore larger iteration budgets. This
screen uses one fit per setting on the same 2023 validation year (never 2024)
to locate a useful region; the finalists are then re-run with the rolling
protocol in ``tuning.py``.

Run from the repository root::

    python team/d-mlp/screen.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_team_benchmark import SNAPSHOT, load_snapshot  # noqa: E402
from team_benchmark_core import metric  # noqa: E402

SUMMARY = ROOT / 'team' / 'd-mlp' / 'summary'
SETTINGS = [
    ('(80,40) alpha=1 iter=400', (80, 40), 1.0, 400, 0.001),
    ('(80,40) alpha=10 iter=400', (80, 40), 10.0, 400, 0.001),
    ('(80,40) alpha=10 iter=1000', (80, 40), 10.0, 1000, 0.001),
    ('(80,40) alpha=10 iter=2000', (80, 40), 10.0, 2000, 0.001),
    ('(80,40) alpha=30 iter=1000', (80, 40), 30.0, 1000, 0.001),
    ('(80,40) alpha=10 iter=1000 lr=0.003', (80, 40), 10.0, 1000, 0.003),
    ('(40,) alpha=10 iter=1000', (40,), 10.0, 1000, 0.001),
    ('(160,80) alpha=10 iter=1000', (160, 80), 10.0, 1000, 0.001),
]


def main():
    features, targets, hourly = load_snapshot(SNAPSHOT)
    train = features.index[features.index < '2023-01-01']
    evaluate = features.index[(features.index >= '2023-01-01') & (features.index < '2024-01-01')]
    rows = []
    for label, hidden, alpha, max_iter, lr in SETTINGS:
        network = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                         learning_rate_init=lr, random_state=42, early_stopping=False),
        )
        model = TransformedTargetRegressor(regressor=network, transformer=StandardScaler())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            started = time.perf_counter()
            model.fit(features.loc[train], targets.loc[train])
            fit_seconds = time.perf_counter() - started
        curves = np.asarray(model.predict(features.loc[evaluate]))
        target = hourly[hourly.day.isin(evaluate)][['day', 'hour', 'DK1']].copy()
        target['prediction'] = (pd.DataFrame(curves, index=evaluate).loc[pd.DatetimeIndex(target.day)]
                                .to_numpy()[np.arange(len(target)), target.hour.to_numpy()])
        score = metric(target.DK1, target.prediction)
        rows.append({'setting': label, 'alpha': alpha, 'max_iter': max_iter, 'learning_rate_init': lr,
                     'fit_seconds': round(fit_seconds, 1),
                     'n_iter': int(np.max(model.regressor_.named_steps['mlpregressor'].n_iter_)),
                     'convergence_warning': int(any(issubclass(w.category, ConvergenceWarning) for w in caught)),
                     'mae': score['mae'], 'rmse': score['rmse'], 'rate_5pct': score['rate_5pct'],
                     'negative_mae': score['negative_mae']})
        print(f'{label}: MAE {score["mae"]:.3f} RMSE {score["rmse"]:.3f} '
              f'fit {fit_seconds:.1f}s', flush=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY / 'screen_2023.csv', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
