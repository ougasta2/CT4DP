"""Find calendar and optional cluster conditions associated with large errors."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from energy_pilot.mining import apriori, mask_for


ZONE = "Europe/Copenhagen"
MIN_SUPPORT = 0.05
MIN_DEVELOPMENT_EVENTS = 20


def load_hours(predictions, regimes):
    frame = pd.read_csv(predictions)
    timestamps = pd.to_datetime(frame["HourUTC"], utc=True)
    expected = pd.date_range(
        pd.Timestamp("2023-01-01", tz=ZONE).tz_convert("UTC"),
        pd.Timestamp("2025-01-01", tz=ZONE).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    frame.index = pd.DatetimeIndex(timestamps)
    frame = frame.sort_index()
    if not frame.index.equals(expected):
        raise ValueError("Run the full 2023–2024 benchmark before analyzing errors.")
    if not np.isfinite(frame[["actual", "prediction"]].to_numpy()).all():
        raise ValueError("Actual prices and predictions must be finite.")

    local = frame.index.tz_convert(ZONE)
    frame["day"] = local.tz_localize(None).normalize()
    frame["year"] = local.year
    frame["season"] = np.array(["winter", "spring", "summer", "autumn"])[
        (local.month % 12) // 3
    ]
    frame["time_of_day"] = np.array(["night", "morning", "afternoon", "evening"])[
        local.hour // 6
    ]
    frame["absolute_error"] = (frame["prediction"] - frame["actual"]).abs()
    if regimes is not None:
        labels = pd.read_csv(regimes, usecols=["day", "cluster"], parse_dates=["day"])
        frame = frame.join(labels.set_index("day"), on="day", validate="many_to_one")
        if frame["cluster"].isna().any():
            raise ValueError("The regimes file must label every evaluation day.")
    return frame


def make_baskets(frame):
    columns = ["season", "time_of_day"]
    if "cluster" in frame:
        columns.append("cluster")
    return [
        frozenset(f"{column}={value}" for column, value in zip(columns, row))
        for row in frame[columns].itertuples(index=False, name=None)
    ]


def rule_metrics(baskets, events, antecedent):
    mask = mask_for(baskets, antecedent)
    hours = int(mask.sum())
    hits = int(events[mask].sum())
    base_rate = float(events.mean())
    confidence = hits / hours if hours else None
    return {
        "hours": hours,
        "large_errors": hits,
        "support": hours / len(baskets),
        "confidence": confidence,
        "base_rate": base_rate,
        "lift": confidence / base_rate if confidence is not None and base_rate else None,
    }


def analyze(frame):
    development = frame[frame["year"] == 2023]
    evaluation = frame[frame["year"] == 2024]
    threshold = development["absolute_error"].quantile(0.90)
    dev_baskets, eval_baskets = make_baskets(development), make_baskets(evaluation)
    dev_events = development["absolute_error"].to_numpy() > threshold
    eval_events = evaluation["absolute_error"].to_numpy() > threshold

    groups = {"calendar": [], "cluster": []}
    for antecedent in apriori(dev_baskets, min_support=MIN_SUPPORT):
        metrics = rule_metrics(dev_baskets, dev_events, antecedent)
        if metrics["large_errors"] < MIN_DEVELOPMENT_EVENTS:
            continue
        group = "cluster" if any(item.startswith("cluster=") for item in antecedent) else "calendar"
        groups[group].append((antecedent, metrics))

    rows = []
    for group, candidates in groups.items():
        if group == "cluster" and "cluster" not in frame:
            continue
        if not candidates:
            rows.append({
                "group": group,
                "antecedent": None,
                "threshold_eur_mwh": threshold,
                "status": "no rule met development support and event-count filters",
            })
            continue
        antecedent, dev = min(
            candidates,
            key=lambda pair: (-pair[1]["lift"], -pair[1]["hours"], sorted(pair[0])),
        )
        evaluated = rule_metrics(eval_baskets, eval_events, antecedent)
        rows.append({
            "group": group,
            "antecedent": " & ".join(sorted(antecedent)),
            "threshold_eur_mwh": threshold,
            "status": "scored",
            **{f"development_{key}": value for key, value in dev.items()},
            **{f"evaluation_{key}": value for key, value in evaluated.items()},
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = ROOT / "results/team/gradient_boosting"
    parser.add_argument("--predictions", type=Path, default=output / "predictions.csv.gz")
    parser.add_argument("--regimes", type=Path, help="Optional CSV from B with day and cluster columns.")
    parser.add_argument("--output", type=Path, default=output / "apriori_rules.csv")
    args = parser.parse_args()
    rows = analyze(load_hours(args.predictions, args.regimes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rule summaries to {args.output}")


if __name__ == "__main__":
    main()
