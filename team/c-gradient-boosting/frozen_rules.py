"""Score the frozen Ridge rules on other forecasts and repeat the afternoon comparison.

    python team/c-gradient-boosting/frozen_rules.py
    python team/c-gradient-boosting/frozen_rules.py --runs gradient_boosting_baseline ridge

For each run, a large error is an absolute error above that run's own 2023
90th percentile (model-specific threshold, derived from 2023 errors only).
The two rules selected on 2023 Ridge errors (autumn + evening and cluster 0 +
afternoon) are scored unchanged on 2023 and 2024. Within 2024 afternoon hours
(12:00-17:59), the large-error rate of cluster 0 is compared with the other
clusters; the 95% percentile interval resamples contiguous seven-day blocks of
2024 delivery days (5,000 draws), keeping all hours of a day together.
Writes results/team/gradient_boosting_summary/frozen_rules.csv and
afternoon_comparison.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_apriori import load_hours
from summarize import OUT, check_runs

REGIMES = HERE.parents[1] / "team" / "b-random-forest" / "temporaryresults" / "regimes.csv"
FROZEN = {  # selected on 2023 Ridge errors (report, Table III)
    "Autumn + evening": lambda f: (f.season == "autumn") & (f.time_of_day == "evening"),
    "Cluster 0 + afternoon": lambda f: (f.cluster == 0) & (f.time_of_day == "afternoon"),
}


def large_errors(frame):
    threshold = frame.loc[frame.year == 2023, "absolute_error"].quantile(0.90)
    return frame.assign(large=frame.absolute_error > threshold), threshold


def frozen_rules(frame, run, threshold):
    rows = []
    for year in (2023, 2024):
        hours = frame[frame.year == year]
        base = hours.large.mean()
        for rule, condition in FROZEN.items():
            match = hours[condition(hours)]
            rows.append({"run": run, "threshold": threshold, "year": year, "rule": rule,
                         "events": int(match.large.sum()), "hours": len(match),
                         "rate": match.large.mean(), "base_rate": base,
                         "lift": match.large.mean() / base})
    return rows


def block_interval(day_stats, draws=5000, block=7, seed=42):
    """Percentile interval for rate(cluster 0) - rate(other) over moving 7-day blocks."""
    rng = np.random.default_rng(seed)
    values = day_stats[["e0", "n0", "e1", "n1"]].to_numpy(float)
    n_days = len(values)
    starts = np.arange(n_days - block + 1)
    n_blocks = int(np.ceil(n_days / block))
    diffs = np.empty(draws)
    for i in range(draws):
        picks = rng.choice(starts, n_blocks)
        idx = (picks[:, None] + np.arange(block)).ravel()[:n_days]
        e0, n0, e1, n1 = values[idx].sum(axis=0)
        diffs[i] = e0 / n0 - e1 / n1
    return np.percentile(diffs, [2.5, 97.5])


def afternoon(frame, run):
    hours = frame[(frame.year == 2024) & (frame.time_of_day == "afternoon")].copy()
    hours["is0"] = hours.cluster == 0
    day = hours.groupby("day").apply(lambda d: pd.Series({
        "e0": d.large[d.is0].sum(), "n0": d.is0.sum(),
        "e1": d.large[~d.is0].sum(), "n1": (~d.is0).sum()}), include_groups=False)
    low, high = block_interval(day)
    zero, rest = hours[hours.is0], hours[~hours.is0]
    return {"run": run,
            "all_events": int(hours.large.sum()), "all_hours": len(hours), "all_rate": hours.large.mean(),
            "c0_events": int(zero.large.sum()), "c0_hours": len(zero), "c0_rate": zero.large.mean(),
            "other_events": int(rest.large.sum()), "other_hours": len(rest), "other_rate": rest.large.mean(),
            "difference_pp": 100 * (zero.large.mean() - rest.large.mean()),
            "ci_low_pp": 100 * low, "ci_high_pp": 100 * high}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["ridge", "gradient_boosting_baseline"])
    parser.add_argument("--regimes", type=Path, default=REGIMES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    found = check_runs(args.runs)
    rules, comparisons = [], []
    for run, path in found.items():
        frame, threshold = large_errors(load_hours(path / "predictions.csv.gz", args.regimes))
        rules += frozen_rules(frame, run, threshold)
        comparisons.append(afternoon(frame, run))
    rules, comparisons = pd.DataFrame(rules), pd.DataFrame(comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    rules.to_csv(args.output / "frozen_rules.csv", index=False)
    comparisons.to_csv(args.output / "afternoon_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(rules.round(4).to_string(index=False))
    print(comparisons.round(4).T.to_string())


if __name__ == "__main__":
    main()
