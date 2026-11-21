"""Compare gradient-boosting runs and list every Apriori rule for 2023 and 2024.

    python team/c-gradient-boosting/summarize.py
    python team/c-gradient-boosting/summarize.py --regimes results/team/random_forest/regimes.csv

Writes results/team/gradient_boosting_summary/{comparison.csv, all_rules_<run>.csv}.
Rules are found and ranked on 2023 only; 2024 columns are read-only checks.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from analyze_apriori import (MIN_DEVELOPMENT_EVENTS, MIN_SUPPORT, apriori, load_hours,
                             make_baskets, rule_metrics)

RESULTS = ROOT / "results" / "team"
OUT = RESULTS / "gradient_boosting_summary"


def runs():
    found = {p.parent.name: p.parent for p in sorted(RESULTS.glob("*/metrics.csv"))}
    return {k: v for k, v in found.items() if (v / "predictions.csv.gz").exists()}


def comparison(found):
    rows = []
    for name, path in found.items():
        metrics = pd.read_csv(path / "metrics.csv")
        timing = pd.read_csv(path / "timing.csv")
        timing["year"] = pd.to_datetime(timing["month"]).dt.year
        seconds = timing.groupby("year")[["fit_seconds", "predict_seconds"]].sum()
        metrics = metrics.join(seconds, on="year")
        metrics.insert(0, "run", name)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def all_rules(frame):
    dev, ev = frame[frame.year == 2023], frame[frame.year == 2024]
    threshold = dev.absolute_error.quantile(0.90)
    dev_b, ev_b = make_baskets(dev), make_baskets(ev)
    dev_e, ev_e = dev.absolute_error.to_numpy() > threshold, ev.absolute_error.to_numpy() > threshold
    rows = []
    for antecedent in apriori(dev_b, min_support=MIN_SUPPORT):
        d = rule_metrics(dev_b, dev_e, antecedent)
        if d["large_errors"] < MIN_DEVELOPMENT_EVENTS:
            continue
        e = rule_metrics(ev_b, ev_e, antecedent)
        rows.append({
            "antecedent": " & ".join(sorted(antecedent)),
            "group": "cluster" if any(i.startswith("cluster=") for i in antecedent) else "calendar",
            "support_2023": d["support"], "confidence_2023": d["confidence"], "lift_2023": d["lift"],
            "large_errors_2023": d["large_errors"],
            "support_2024": e["support"], "confidence_2024": e["confidence"], "lift_2024": e["lift"],
            "large_errors_2024": e["large_errors"],
        })
    table = pd.DataFrame(rows).sort_values("lift_2023", ascending=False)
    table.attrs["threshold"] = threshold
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    found = runs()
    table = comparison(found)
    table.to_csv(OUT / "comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(table[["run", "year", "mae", "rmse", "hits_5pct", "eligible", "rate_5pct",
                 "negative_hours", "negative_mae", "fit_seconds"]].round(3).to_string(index=False))
    for name, path in found.items():
        rules = all_rules(load_hours(path / "predictions.csv.gz", args.regimes))
        rules.to_csv(OUT / f"all_rules_{name}.csv", index=False)
        lifted = rules[rules.lift_2023 > 1.2]
        held = (lifted.lift_2024 > 1).sum()
        print(f"\n{name}: threshold {rules.attrs['threshold']:.1f} EUR/MWh, {len(rules)} rules, "
              f"{len(lifted)} with 2023 lift > 1.2, {held} of those still > 1 in 2024")
        print(rules.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
