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
import train_dk1_preauction as base
import train_dk1_weather_ablation as weather_source

NAMES = ('ridge', 'lasso', 'random_forest', 'mlp')
OUT = ROOT / 'results' / 'team'

def prepare():
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

def run(name, start, end, min_train_days=365):
    build = importlib.import_module(f'models.{name}.model').build_model
    return run_prepared(name, build, start, end, *prepare(), OUT / name, min_train_days)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=NAMES, required=True)
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-01-01', help='exclusive, first day of a month')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.model, args.start, args.end)

if __name__ == '__main__':
    main()
