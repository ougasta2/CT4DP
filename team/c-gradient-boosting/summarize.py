"""Compare gradient-boosting runs and list every Apriori rule for 2023 and 2024.

    python team/c-gradient-boosting/summarize.py
    python team/c-gradient-boosting/summarize.py --regimes team/b-random-forest/temporaryresults/regimes.csv
    python team/c-gradient-boosting/summarize.py --runs gradient_boosting_baseline ridge

By default the summary covers the starter, the three one-change variants and
the Ridge baseline. --runs takes run names (folders under results/team/) or
paths to run folders. Every selected run must cover all hours of 2023–2024;
the script checks this before writing anything and says what to rerun.

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
from analyze_apriori import (MIN_DEVELOPMENT_EVENTS, MIN_SUPPORT, ZONE, apriori, load_hours,
                             make_baskets, rule_metrics)
from run_experiments import VARIANTS

RESULTS = ROOT / "results" / "team"
OUT = RESULTS / "gradient_boosting_summary"
YEARS = (2023, 2024)
EXPECTED_HOURS = pd.date_range(
    pd.Timestamp("2023-01-01", tz=ZONE).tz_convert("UTC"),
    pd.Timestamp("2025-01-01", tz=ZONE).tz_convert("UTC"),
    freq="h",
    inclusive="left",
)

# Default runs and the command that produces each one for the full period.
RERUN = {
    "gradient_boosting_baseline": "python scripts/run_team_benchmark.py --model gradient_boosting "
                                  "--data --output results/team/gradient_boosting_baseline",
    **{f"gradient_boosting_{v}": f"python team/c-gradient-boosting/run_experiments.py {v}"
       for v in VARIANTS},
    "ridge": "python scripts/run_team_benchmark.py --model ridge --data",
}
DEFAULT_RUNS = list(RERUN)


def resolve(run):
    """Accept a run name under results/team/ or a path to a run folder."""
    path = Path(run)
    if not path.is_absolute() and len(path.parts) == 1:
        path = RESULTS / run
    return path.name, path


def coverage_problem(path):
    """Return None if the run covers all 2023–2024 hours, else a short reason."""
    missing = [f for f in ("metrics.csv", "timing.csv", "predictions.csv.gz") if not (path / f).exists()]
    if missing:
        return f"missing {', '.join(missing)} in {path}"
    stamps = pd.to_datetime(pd.read_csv(path / "predictions.csv.gz", usecols=["HourUTC"])["HourUTC"], utc=True)
    index = pd.DatetimeIndex(stamps).sort_values()
    if not index.equals(EXPECTED_HOURS):
        if len(index) == 0:
            return "predictions.csv.gz is empty"
        first, last = index[0].tz_convert(ZONE), index[-1].tz_convert(ZONE)
        return (f"predictions cover {len(index):,} hours ({first:%Y-%m-%d} to {last:%Y-%m-%d}), "
                f"expected {len(EXPECTED_HOURS):,} hours (2023-01-01 to 2024-12-31)")
    years = set(pd.read_csv(path / "metrics.csv")["year"])
    if not set(YEARS) <= years:
        return f"metrics.csv has years {sorted(years)}, expected {list(YEARS)}"
    return None


def check_runs(runs):
    """Resolve the selected runs and stop with a clear message if any is incomplete."""
    found, problems = {}, []
    for run in runs:
        name, path = resolve(run)
        problem = coverage_problem(path)
        if problem:
            fix = RERUN.get(name, "rerun it over the full period (--start 2023-01-01 --end 2025-01-01)")
            problems.append(f"  {name}: {problem}\n    rerun: {fix}")
        found[name] = path
    if problems:
        raise SystemExit("Not summarizing: these runs do not cover all of 2023–2024. "
                         "Nothing was written.\n" + "\n".join(problems))
    return found


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="run names under results/team/ or run folders (default: %(default)s)")
    parser.add_argument("--regimes", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    found = check_runs(args.runs)            # stops here, before writing, if a run is incomplete
    table = comparison(found)
    rules = {name: all_rules(load_hours(path / "predictions.csv.gz", args.regimes))
             for name, path in found.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(table[["run", "year", "mae", "rmse", "hits_5pct", "eligible", "rate_5pct",
                 "negative_hours", "negative_mae", "fit_seconds"]].round(3).to_string(index=False))
    for name, frame in rules.items():
        frame.to_csv(args.output / f"all_rules_{name}.csv", index=False)
        lifted = frame[frame.lift_2023 > 1.2]
        held = (lifted.lift_2024 > 1).sum()
        print(f"\n{name}: threshold {frame.attrs['threshold']:.2f} EUR/MWh, {len(frame)} rules, "
              f"{len(lifted)} with 2023 lift > 1.2, {held} of those still > 1 in 2024")
        print(frame.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
