"""Run this repository's model on the project lead's frozen data snapshot."""

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from model import build_model
from team_benchmark_core import run_prepared


def load_inputs(data):
    def read_table(name, index):
        return pd.read_csv(data / f"{name}.csv.gz", index_col=index,
                           parse_dates=[index], float_precision="round_trip")

    features = read_table("features", "day")
    targets = read_table("targets", "day")
    hourly = read_table("hourly", "HourUTC")
    hourly["day"] = pd.to_datetime(hourly["day"])
    if not features.index.equals(targets.index) or list(targets.columns) != list(map(str, range(24))):
        raise ValueError("Expected matching feature/target days and 24 ordered target hours")
    return features, targets, hourly


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01", help="Exclusive, first day of a month")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    required = ("manifest.json", "features.csv.gz", "targets.csv.gz", "hourly.csv.gz")
    missing = [name for name in required if not (args.data / name).is_file()]
    if missing:
        parser.error("Obtain the local benchmark data snapshot from the project lead. Missing: " + ", ".join(missing))
    with threadpool_limits(limits=2):
        run_prepared(root.name, build_model, args.start, args.end, *load_inputs(args.data), args.output)


if __name__ == "__main__":
    main()
