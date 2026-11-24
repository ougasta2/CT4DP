"""Diagnostics for the MLP workspace: per-month, per-hour and paired errors.

Reads the aligned prediction files written by the shared benchmark
(``results/team/<model>/predictions.csv.gz``) and compares a neural network
with the frozen Ridge baseline hour by hour, so the annual metrics can be
broken down and the month-to-month stability of any gain can be checked.

The uncertainty of the MAE difference is estimated three ways, because hourly
errors are strongly autocorrelated: an i.i.d. bootstrap over delivery days, a
moving-block bootstrap over 7 and 14 consecutive days, and a Newey-West
interval on the daily error differences. The block and Newey-West intervals are
the ones to quote; the i.i.d. one ignores dependence between neighbouring days
and is kept for comparison only.

Run from the repository root after the three benchmark runs::

    python team/d-mlp/starter.py
    python scripts/run_team_benchmark.py --model ridge --data
    python scripts/run_team_benchmark.py --model mlp --data
    python team/d-mlp/analysis.py

Writes ``team/d-mlp/summary/per_month_mae.csv``,
``team/d-mlp/summary/per_hour_mae.csv``, ``team/d-mlp/summary/paired_bootstrap.csv``
and ``team/d-mlp/summary/mlp_diagnostics.png``.
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
BLOCKS = (7, 14)
NW_LAGS = 7


def load(model):
    frame = pd.read_csv(RESULTS / model / 'predictions.csv.gz', parse_dates=['HourUTC', 'day'],
                        float_precision='round_trip')
    frame = frame.set_index('HourUTC').sort_index()
    frame['month'] = frame['day'].dt.to_period('M').astype(str)
    frame['abs_error'] = (frame['prediction'] - frame['actual']).abs()
    return frame[['day', 'hour', 'actual', 'prediction', 'abs_error', 'month']]


def day_sums(reference, candidate, year):
    """Absolute-error totals per delivery day, so days stay the resampling unit."""
    ref = reference[reference.day.dt.year == year]
    cand = candidate[candidate.day.dt.year == year]
    joined = ref[['day', 'abs_error']].join(cand[['abs_error']], lsuffix='_ref', rsuffix='_cand', how='inner')
    if len(joined) != len(ref) or len(joined) != len(cand):
        raise ValueError(f'misaligned hours in {year}: {len(joined)}, {len(ref)}, {len(cand)}')
    totals = joined.groupby('day')[['abs_error_ref', 'abs_error_cand']].sum().to_numpy()
    return totals, joined.groupby('day').size().to_numpy()


def iid_bootstrap(totals, hours, replicates=REPLICATES, seed=SEED):
    """Resample delivery days independently (ignores dependence between days)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hours), size=(replicates, len(hours)))
    return (totals[draws, 1] - totals[draws, 0]).sum(axis=1) / hours[draws].sum(axis=1)


def block_bootstrap(totals, hours, block, replicates=REPLICATES, seed=SEED):
    """Moving-block bootstrap: resample runs of ``block`` consecutive delivery days."""
    days = len(hours)
    rng = np.random.default_rng(seed + block)
    starts = rng.integers(0, days, size=(replicates, int(np.ceil(days / block))))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]) % days
    index = index.reshape(replicates, -1)[:, :days]
    return (totals[index, 1] - totals[index, 0]).sum(axis=1) / hours[index].sum(axis=1)


def newey_west_interval(totals, hours, observed, lags=NW_LAGS):
    """Newey-West standard error of the daily MAE difference (Bartlett weights)."""
    daily = (totals[:, 1] - totals[:, 0]) / hours
    centred = daily - daily.mean()
    n = len(daily)
    variance = centred @ centred / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (centred[lag:] @ centred[:-lag]) / n
    se = np.sqrt(max(variance, 0) / n)
    return observed - 1.96 * se, observed + 1.96 * se, se


def compare(reference, candidate, year, replicates=REPLICATES, seed=SEED):
    """MAE difference (candidate - reference) with dependence-aware intervals."""
    totals, hours = day_sums(reference, candidate, year)
    observed = (totals[:, 1].sum() - totals[:, 0].sum()) / hours.sum()
    iid = iid_bootstrap(totals, hours, replicates, seed)
    row = {'year': year, 'hours': int(hours.sum()), 'days': int(len(hours)),
           'mae_reference': float(totals[:, 0].sum() / hours.sum()),
           'mae_candidate': float(totals[:, 1].sum() / hours.sum()),
           'mae_difference': float(observed),
           'ci_low_iid': float(np.percentile(iid, 2.5)), 'ci_high_iid': float(np.percentile(iid, 97.5)),
           'share_of_draws_positive': float((iid > 0).mean())}
    for block in BLOCKS:
        draws = block_bootstrap(totals, hours, block, replicates, seed)
        row[f'ci_low_block{block}'] = float(np.percentile(draws, 2.5))
        row[f'ci_high_block{block}'] = float(np.percentile(draws, 97.5))
    low, high, se = newey_west_interval(totals, hours, observed)
    row.update({'ci_low_nw': float(low), 'ci_high_nw': float(high), 'nw_standard_error': float(se)})
    return row


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

    rows = [compare(frames[BASELINE], frames[name], year)
            for name in MODELS for year in (2023, 2024)]
    bootstrap = pd.DataFrame(rows).assign(model=[name for name in MODELS for _ in (2023, 2024)])
    bootstrap['months_candidate_better'] = [
        int((months.loc[months.index.str.startswith(str(year)), name]
             < months.loc[months.index.str.startswith(str(year)), BASELINE]).sum())
        for name in MODELS for year in (2023, 2024)]
    bootstrap.to_csv(SUMMARY / 'paired_bootstrap.csv', index=False)

    pd.set_option('display.width', 200)
    print(months.round(3).to_string())
    columns = ['model', 'year', 'mae_reference', 'mae_candidate', 'mae_difference',
               'ci_low_iid', 'ci_high_iid', 'ci_low_block7', 'ci_high_block7',
               'ci_low_block14', 'ci_high_block14', 'ci_low_nw', 'ci_high_nw']
    print(bootstrap[columns].round(3).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
