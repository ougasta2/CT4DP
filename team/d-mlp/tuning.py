"""Rolling validation over 2023 for the MLP workspace (option D).

The shared benchmark only reports scores for the months it evaluates, so
settings are chosen here instead: every month of 2023 is predicted from a
model fitted on all earlier days only (expanding window, same as the
benchmark), and 2024 is never touched. Scoring uses the shared
``team_benchmark_core.metric`` on the hourly observations, so validation
numbers are on exactly the same scale as the final benchmark metrics.

Run from the repository root::

    python team/d-mlp/tuning.py --stage 1   # sizes and L2 around the starter
    python team/d-mlp/tuning.py --stage 2   # combinations that stage 1 suggested

Writes ``team/d-mlp/summary/validation_grid_stage<N>.csv`` (one row per
setting) and ``team/d-mlp/summary/validation_months_stage<N>.csv`` (one row
per setting-month).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_team_benchmark import SNAPSHOT, load_snapshot  # noqa: E402
from team_benchmark_core import metric  # noqa: E402

SUMMARY = ROOT / 'team' / 'd-mlp' / 'summary'

# Stage 1: one change at a time from the starter (80, 40), alpha=1.0, 400
# iterations. ``ridge`` is the frozen baseline on the same validation months.
STAGE1 = [
    ('ridge alpha=100', None, None, None),
    ('starter (80,40) alpha=1 iter=400', (80, 40), 1.0, 400),
    ('small (40,) alpha=1 iter=400', (40,), 1.0, 400),
    ('large (160,80) alpha=1 iter=400', (160, 80), 1.0, 400),
    ('alpha=0.1 iter=400', (80, 40), 0.1, 400),
    ('alpha=10 iter=400', (80, 40), 10.0, 400),
    ('alpha=100 iter=400', (80, 40), 100.0, 400),
    ('starter iter=800', (80, 40), 1.0, 800),
]

# Stage 2: stronger L2 combined with the smaller and larger networks, after
# stage 1 showed alpha=10 helping and capacity not helping.
STAGE2 = [
    ('(40,) alpha=10', (40,), 10.0, 400),
    ('(40,) alpha=30', (40,), 30.0, 400),
    ('(80,) alpha=10', (80,), 10.0, 400),
    ('(80,40) alpha=30', (80, 40), 30.0, 400),
    ('(160,80) alpha=10', (160, 80), 10.0, 400),
]
# Stage 3: narrow the region stage 2 found (small network, alpha=10), check
# the iteration budget of the chosen setting and one alternative target scale.
STAGE3 = [
    ('(40,) alpha=5', (40,), 5.0, 400, 'standard'),
    ('(32,) alpha=10', (32,), 10.0, 400, 'standard'),
    ('(24,) alpha=10', (24,), 10.0, 400, 'standard'),
    ('(40,) alpha=20', (40,), 20.0, 400, 'standard'),
    ('(40,) alpha=10 iter=800', (40,), 10.0, 800, 'standard'),
    ('(40,) alpha=10 quantile target', (40,), 10.0, 400, 'quantile'),
]
STAGES = {1: STAGE1, 2: STAGE2, 3: STAGE3}

MONTHS = pd.date_range('2023-01-01', '2024-01-01', freq='MS', inclusive='left')


def build(hidden, alpha, max_iter, target='standard'):
    if hidden is None:
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    network = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                     random_state=42, early_stopping=False),
    )
    if target == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal',
                                          n_quantiles=100, random_state=42)
    else:
        transformer = StandardScaler()
    return TransformedTargetRegressor(regressor=network, transformer=transformer)


def hourly_score(curves, evaluate, hourly):
    """Score a (days, 24) curve matrix on the shared hourly observations."""
    target = hourly[hourly.day.isin(evaluate)][['day', 'hour', 'DK1']].copy()
    target['prediction'] = (
        pd.DataFrame(curves, index=evaluate).loc[pd.DatetimeIndex(target.day)]
        .to_numpy()[np.arange(len(target)), target.hour.to_numpy()]
    )
    return metric(target.DK1, target.prediction)


def main(stage):
    settings = [item if len(item) == 5 else (*item, 'standard') for item in STAGES[stage]]
    features, targets, hourly = load_snapshot(SNAPSHOT)
    rows, months_by_setting = [], {}
    for label, hidden, alpha, max_iter, target in settings:
        month_rows, warned, iterations, fit_total = [], 0, [], 0.0
        for month in MONTHS:
            next_month = month + pd.offsets.MonthBegin(1)
            train = features.index[features.index < month]
            evaluate = features.index[(features.index >= month) & (features.index < next_month)]
            model = build(hidden, alpha, max_iter, target)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                started = time.perf_counter()
                model.fit(features.loc[train], targets.loc[train])
                fit_seconds = time.perf_counter() - started
            warned += int(any(issubclass(w.category, ConvergenceWarning) for w in caught))
            if hidden is not None:
                used = model.regressor_.named_steps['mlpregressor'].n_iter_
                iterations.append(int(np.max(used)))  # int in new sklearn, array in old
            curves = np.asarray(model.predict(features.loc[evaluate]))
            score = hourly_score(curves, evaluate, hourly)
            score.update({'month': str(month.date()), 'fit_seconds': fit_seconds})
            month_rows.append(score)
            fit_total += fit_seconds
        months = pd.DataFrame(month_rows)
        months_by_setting[label] = months
        pooled_score = {
            'mae': float(np.average(months.mae, weights=months.hours)),
            'rmse': float(np.sqrt(np.average(months.rmse ** 2, weights=months.hours))),
            'rate_5pct': float(months.hits_5pct.sum() / months.eligible.sum()),
            'hits_5pct': int(months.hits_5pct.sum()),
            'eligible': int(months.eligible.sum()),
            'negative_hours': int(months.negative_hours.sum()),
        }
        rows.append({'setting': label,
                     'hidden': None if hidden is None else '-'.join(map(str, hidden)),
                     'alpha': alpha, 'max_iter': max_iter, 'target_transform': target,
                     'hours': int(months.hours.sum()),
                     **pooled_score,
                     'fit_seconds_12_refits': round(fit_total, 1),
                     'mean_n_iter': float(np.mean(iterations)) if iterations else None,
                     'months_with_convergence_warning': warned})
        print(f'{label}: MAE {pooled_score["mae"]:.3f} RMSE {pooled_score["rmse"]:.3f} '
              f'hit {pooled_score["rate_5pct"]:.4f} fit {fit_total:.1f}s', flush=True)

    SUMMARY.mkdir(parents=True, exist_ok=True)
    suffix = f'_stage{stage}'
    all_months = pd.concat([frame.assign(setting=label) for label, frame in months_by_setting.items()])
    all_months.to_csv(SUMMARY / f'validation_months{suffix}.csv', index=False)

    grid = pd.DataFrame(rows)
    if 'ridge alpha=100' in months_by_setting:
        baseline = grid.set_index('setting').loc['ridge alpha=100']
        ridge_months = months_by_setting['ridge alpha=100'].set_index('month').mae
    else:
        # Stage 2 reuses the baseline scored in stage 1 on the same months.
        previous = pd.read_csv(SUMMARY / 'validation_grid_stage1.csv').set_index('setting')
        baseline = previous.loc['ridge alpha=100']
        ridge_months = (pd.read_csv(SUMMARY / 'validation_months_stage1.csv')
                        .query('setting == "ridge alpha=100"').set_index('month').mae)
    mask = grid.setting != 'ridge alpha=100'
    grid.loc[mask, 'mae_vs_ridge'] = grid.loc[mask, 'mae'] - baseline.mae
    grid.loc[mask, 'rmse_vs_ridge'] = grid.loc[mask, 'rmse'] - baseline.rmse
    grid.loc[mask, 'months_worse_than_ridge'] = [
        int((months_by_setting[label].set_index('month').mae > ridge_months).sum())
        for label in grid.loc[mask, 'setting']
    ]
    grid.to_csv(SUMMARY / f'validation_grid{suffix}.csv', index=False)
    print(grid.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=sorted(STAGES), default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.stage)
