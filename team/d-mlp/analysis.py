"""Diagnostics for the MLP workspace: per-month, per-hour and paired errors.

Reads the aligned prediction files written by the shared benchmark
(``results/team/<model>/predictions.csv.gz``) and compares a neural network
with the frozen Ridge baseline hour by hour, so the annual metrics can be
broken down and the month-to-month stability of any gain can be checked.

Run from the repository root after
``python scripts/run_team_benchmark.py --model mlp --data``::

    python team/d-mlp/analysis.py

Writes ``team/d-mlp/summary/per_month_mae.csv``,
``team/d-mlp/summary/per_hour_mae.csv`` and
``team/d-mlp/summary/paired_bootstrap.csv``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'team'
SUMMARY = ROOT / 'team' / 'd-mlp' / 'summary'
BASELINE = 'ridge'
MODELS = ('mlp', 'mlp-starter')
REPLICATES = 2000
SEED = 42


def load(model):
    frame = pd.read_csv(RESULTS / model / 'predictions.csv.gz', parse_dates=['HourUTC', 'day'],
                        float_precision='round_trip')
    frame = frame.set_index('HourUTC').sort_index()
    frame['month'] = frame['day'].dt.to_period('M').astype(str)
    frame['abs_error'] = (frame['prediction'] - frame['actual']).abs()
    return frame[['day', 'hour', 'actual', 'prediction', 'abs_error', 'month']]


def paired_bootstrap(reference, candidate, year, replicates=REPLICATES, seed=SEED):
    """95% interval for MAE(candidate) - MAE(reference), resampling days."""
    ref = reference[reference.day.dt.year == year]
    cand = candidate[candidate.day.dt.year == year]
    joined = ref[['day', 'abs_error']].join(cand[['abs_error']], lsuffix='_ref', rsuffix='_cand', how='inner')
    if len(joined) != len(ref) or len(joined) != len(cand):
        raise ValueError(f'misaligned hours in {year}: {len(joined)}, {len(ref)}, {len(cand)}')
    days = joined.day.unique()
    by_day = joined.groupby('day')[['abs_error_ref', 'abs_error_cand']].sum()
    counts = joined.groupby('day').size()
    to_ref, to_cand = by_day['abs_error_ref'].to_numpy(), by_day['abs_error_cand'].to_numpy()
    hours = counts.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(replicates, len(days)))
    diff = ((to_cand[draws] - to_ref[draws]).sum(axis=1) / hours[draws].sum(axis=1))
    observed = (to_cand.sum() - to_ref.sum()) / hours.sum()
    return {'year': year, 'hours': int(hours.sum()), 'days': int(len(days)),
            'mae_reference': float(to_ref.sum() / hours.sum()),
            'mae_candidate': float(to_cand.sum() / hours.sum()),
            'mae_difference': float(observed),
            'ci_low': float(np.percentile(diff, 2.5)), 'ci_high': float(np.percentile(diff, 97.5)),
            'share_of_draws_positive': float((diff > 0).mean())}


def plot(months, hours, dest):
    """Two panels for the write-up: monthly MAE and MAE by delivery hour."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(months))
    labels = {BASELINE: 'Ridge alpha=100', 'mlp': 'MLP (40,), alpha=10', 'mlp-starter': 'MLP starter'}
    for name, style in ((BASELINE, ('#444444', '--')), ('mlp', ('#1b6ca8', '-')),
                        ('mlp-starter', ('#c46a00', ':'))):
        left.plot(x, months[name], style[1], color=style[0], linewidth=2, label=labels[name])
    left.set_xticks(x[::2])
    left.set_xticklabels([m[2:] for m in months.index[::2]], rotation=90, fontsize=7)
    left.set_xlabel('delivery month')
    left.set_ylabel('MAE (EUR/MWh)')
    left.set_title('Monthly MAE, all hours')
    left.legend(fontsize=8, frameon=False)

    width = 0.4
    right.bar(hours.index - width / 2, hours[BASELINE], width, color='#444444', label='Ridge alpha=100')
    right.bar(hours.index + width / 2, hours['mlp'], width, color='#1b6ca8', label='MLP (40,), alpha=10')
    right.set_xlabel('local delivery hour')
    right.set_ylabel('MAE (EUR/MWh)')
    right.set_title('MAE by delivery hour, 2023-2024')
    right.set_xticks(hours.index[::2])
    right.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def main():
    frames = {name: load(name) for name in (BASELINE, *MODELS)}
    SUMMARY.mkdir(parents=True, exist_ok=True)

    months = pd.concat({name: frame.groupby('month').abs_error.mean()
                        for name, frame in frames.items()}, axis=1)
    months.to_csv(SUMMARY / 'per_month_mae.csv', index_label='month')

    hours = pd.concat({name: frame.groupby('hour').abs_error.mean()
                       for name, frame in frames.items()}, axis=1)
    hours.to_csv(SUMMARY / 'per_hour_mae.csv', index_label='hour')
    plot(months, hours, SUMMARY / 'mlp_diagnostics.png')

    rows = [paired_bootstrap(frames[BASELINE], frames[name], year)
            for name in MODELS for year in (2023, 2024)]
    bootstrap = pd.DataFrame(rows).assign(model=[name for name in MODELS for _ in (2023, 2024)])
    bootstrap['months_candidate_better'] = [
        int((months.loc[months.index.str.startswith(str(year)), name]
             < months.loc[months.index.str.startswith(str(year)), BASELINE]).sum())
        for name in MODELS for year in (2023, 2024)]
    bootstrap.to_csv(SUMMARY / 'paired_bootstrap.csv', index=False)

    pd.set_option('display.width', 120)
    print(months.round(3).to_string())
    print(bootstrap.round(3).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
