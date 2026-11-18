"""Report Random Forest forecast errors by market regime (cluster) and year."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(predictions, regimes, output):
    hours = pd.read_csv(predictions, parse_dates=["day"])
    labels = pd.read_csv(regimes, parse_dates=["day"])
    # Each hourly row gets the label of its local delivery day.
    hours = hours.merge(labels, on="day", how="left", validate="many_to_one")
    if hours["cluster"].isna().any():
        raise ValueError("The regimes file must label every evaluation day")
    hours["error"] = hours["prediction"] - hours["actual"]
    hours["abs_error"] = hours["error"].abs()
    hours["year"] = hours["day"].dt.year
    # Same large-error rule as C's Apriori analysis: 2023 90th percentile.
    threshold = hours.loc[hours["year"] == 2023, "abs_error"].quantile(0.90)
    hours["large_error"] = hours["abs_error"] > threshold
    table = hours.groupby(["year", "cluster"]).agg(
        days=("day", "nunique"),
        hours=("abs_error", "size"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda e: np.sqrt(np.mean(e**2))),
        large_errors=("large_error", "sum"),
        large_error_rate=("large_error", "mean"),
    )
    table.to_csv(output)
    print(f"Large-error threshold (2023 90th percentile): {threshold:.2f} EUR/MWh")
    print(table.round(3).to_string())


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/team/random_forest"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=out / "predictions.csv.gz")
    parser.add_argument("--regimes", type=Path, default=out / "regimes.csv")
    parser.add_argument("--output", type=Path, default=out / "regime_errors.csv")
    args = parser.parse_args()
    run(args.predictions, args.regimes, args.output)


if __name__ == "__main__":
    main()
