"""Common monthly walk-forward benchmark for the four team model workspaces.

Run from the repository root. Existing legacy inputs and results are not changed.
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd
from threadpoolctl import threadpool_limits
from team_benchmark_core import run_prepared

WORKSPACES = {
    'lasso': 'a-lasso',
    'random_forest': 'b-random-forest',
    'gradient_boosting': 'c-gradient-boosting',
    'mlp': 'd-mlp',
}
NAMES = ('ridge', *WORKSPACES)
OUT = ROOT / 'results' / 'team'
SNAPSHOT = ROOT / 'team' / 'data'

def prepare():
    import train_dk1_preauction as base
    import train_dk1_weather_ablation as weather_source

    prices = base.load_prices()
    x, y, hourly = base.make_daily(prices)
    weather, _ = weather_source.weather_curves(prices)
    x = pd.concat([x, weather], axis=1)
    counts = hourly.groupby('day').size().reindex(x.index)
    expected = pd.Series([(pd.Timestamp(d + pd.Timedelta(days=1), tz='Europe/Copenhagen').tz_convert('UTC')
                           - pd.Timestamp(d, tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds() / 3600
                          for d in x.index], index=x.index)
    valid = x.notna().all(axis=1) & y.notna().all(axis=1) & counts.eq(expected)
    return x.loc[valid], y.loc[valid], hourly

def build_model(name):
    module = 'models.ridge.model' if name == 'ridge' else f'team.{WORKSPACES[name]}.model'
    return importlib.import_module(module).build_model()


def load_snapshot(data):
    def read_table(name, index):
        return pd.read_csv(data / f'{name}.csv.gz', index_col=index,
                           parse_dates=[index], float_precision='round_trip')

    features = read_table('features', 'day')
    targets = read_table('targets', 'day')
    hourly = read_table('hourly', 'HourUTC')
    hourly['day'] = pd.to_datetime(hourly['day'])
    if not features.index.equals(targets.index) or list(targets.columns) != list(map(str, range(24))):
        raise ValueError('Expected matching feature/target days and 24 ordered target hours')
    return features, targets, hourly


def run(name, start, end, min_train_days=365, data=None, output=None):
    inputs = prepare() if data is None else load_snapshot(data)
    return run_prepared(name, lambda: build_model(name), start, end, *inputs,
                        output or OUT / name, min_train_days)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=NAMES, required=True)
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-01-01', help='exclusive, first day of a month')
    parser.add_argument('--data', type=Path, nargs='?', const=SNAPSHOT,
                        help='use the bundled snapshot, or give a snapshot directory')
    parser.add_argument('--output', type=Path, help='override results/team/<model>')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.model, args.start, args.end, data=args.data, output=args.output)

if __name__ == '__main__':
    main()
