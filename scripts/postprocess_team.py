"""Compare aligned DK1 forecasts, regime-weighted blends and frozen error rules."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from team_benchmark_core import metric


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "team/data/hourly.csv.gz"
MEMBERS = ["lasso", "random_forest", "gradient_boosting", "mlp"]
ZONE = "Europe/Copenhagen"
BLOCK_DAYS = (3, 7, 14)
DRAWS = 5000
SEED = 42
PRIOR_HOURS = 30 * 24


def aligned_inputs(files, regimes):
    utc = pd.date_range(
        pd.Timestamp("2023-01-01", tz=ZONE).tz_convert("UTC"),
        pd.Timestamp("2025-01-01", tz=ZONE).tz_convert("UTC"),
        freq="h",
        inclusive="left",
        name="HourUTC",
    )
    snapshot = pd.read_csv(SNAPSHOT, usecols=["HourUTC", "DK1"])
    snapshot["HourUTC"] = pd.to_datetime(snapshot["HourUTC"], utc=True)
    actual = snapshot.set_index("HourUTC", verify_integrity=True)["DK1"].reindex(utc)
    local = utc.tz_convert(ZONE)
    frame = pd.DataFrame(
        {"day": local.tz_localize(None).normalize(), "hour": local.hour, "actual": actual},
        index=utc,
    )
    if not np.isfinite(frame.actual).all():
        raise ValueError("Incomplete benchmark targets")
    for name, path in files.items():
        prediction = pd.read_csv(path, parse_dates=["day"])
        prediction["HourUTC"] = pd.to_datetime(prediction["HourUTC"], utc=True)
        prediction = prediction.set_index("HourUTC").sort_index()
        if (
            not prediction.index.equals(utc)
            or not prediction.day.equals(frame.day)
            or not np.array_equal(prediction.hour, frame.hour)
            or not np.allclose(prediction.actual, frame.actual, rtol=0, atol=1e-10)
            or not np.isfinite(prediction.prediction).all()
        ):
            raise ValueError(f"{name}: predictions do not match the complete benchmark")
        frame[name] = prediction.prediction
    labels = pd.read_csv(regimes, usecols=["day", "cluster"], parse_dates=["day"])
    if set(labels.cluster) != {0, 1, 2, 3}:
        raise ValueError("Expected labels from the frozen four-cluster K-means model")
    frame = frame.join(labels.set_index("day"), on="day", validate="many_to_one")
    if frame.cluster.isna().any():
        raise ValueError("Missing market regime for a benchmark day")
    frame["year"] = local.year
    frame["season"] = np.array(["winter", "spring", "summer", "autumn"])[
        (local.month % 12) // 3
    ]
    frame["time_of_day"] = np.array(["night", "morning", "afternoon", "evening"])[
        local.hour // 6
    ]
    return frame


def add_blends(frame):
    development = frame.loc[frame.year.eq(2023)]
    errors = development[MEMBERS].sub(development.actual, axis=0).abs()
    global_mae = errors.mean()
    grouped = errors.groupby(development.cluster)
    counts = grouped.size()
    # Give smaller regimes more weight from the pooled 2023 error estimate.
    regime_mae = (grouped.sum() + PRIOR_HOURS * global_mae).div(counts + PRIOR_HOURS, axis=0)
    inverses = 1 / regime_mae.clip(lower=1e-12)
    regime_weights = inverses.div(inverses.sum(axis=1), axis=0)
    global_weights = 1 / global_mae.clip(lower=1e-12)
    global_weights /= global_weights.sum()
    frame["equal_blend"] = frame[MEMBERS].mean(axis=1)
    frame["global_blend"] = frame[MEMBERS] @ global_weights
    frame["regime_blend"] = (
        frame[MEMBERS].to_numpy() * regime_weights.loc[frame.cluster, MEMBERS].to_numpy()
    ).sum(axis=1)
    weights = pd.concat(
        [global_weights.to_frame().T.rename(index={0: "global"}), regime_weights]
    )
    weights.index.name = "regime"
    weights.insert(0, "calibration_hours", pd.concat([pd.Series({"global": len(errors)}), counts]))
    return weights, global_mae.idxmin()


def block_samples(days, block_lengths=BLOCK_DAYS):
    rng = np.random.default_rng(SEED)
    for block in block_lengths:
        starts = rng.integers(0, days - block + 1, size=(DRAWS, int(np.ceil(days / block))))
        yield block, (starts[:, :, None] + np.arange(block)).reshape(DRAWS, -1)[:, :days]


def paired_mae(frame, best_2023):
    test = frame.loc[frame.year.eq(2024)]
    cols = ["ridge", best_2023, "equal_blend", "global_blend", "regime_blend"]
    errors = test[cols].sub(test.actual, axis=0).abs()
    daily = errors.assign(day=test.day, hours=1).groupby("day").sum()
    contrasts = [
        ("equal_blend", "ridge"),
        ("regime_blend", best_2023),
        ("regime_blend", "equal_blend"),
        ("regime_blend", "global_blend"),
    ]
    rows = []
    for block, indices in block_samples(len(daily)):
        totals = daily.to_numpy()[indices].sum(axis=1)
        for model, baseline in contrasts:
            delta = totals[:, daily.columns.get_loc(model)] - totals[:, daily.columns.get_loc(baseline)]
            low, high = np.quantile(delta / totals[:, daily.columns.get_loc("hours")], [0.025, 0.975])
            rows.append({"model": model, "baseline": baseline, "block_days": block,
                         "delta_mae": (errors[model] - errors[baseline]).mean(),
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def ridge_rule_contrasts(frame):
    """Compare rates within afternoons and within evenings on the same day blocks."""
    error = frame.ridge.sub(frame.actual).abs()
    threshold = error.loc[frame.year.eq(2023)].quantile(0.90)
    test = frame.loc[frame.year.eq(2024)]
    event = error.loc[test.index].gt(threshold)
    afternoon = test.time_of_day.eq("afternoon")
    evening = test.time_of_day.eq("evening")
    cluster_zero = test.cluster.eq(0)
    autumn = test.season.eq("autumn")
    contrasts = [
        ("cluster_0_vs_other_afternoons", afternoon & cluster_zero, afternoon & ~cluster_zero),
        ("autumn_vs_other_evenings", evening & autumn, evening & ~autumn),
    ]
    _, indices = next(block_samples(test.day.nunique(), (7,)))
    rows = []
    for name, matched, control in contrasts:
        daily = pd.DataFrame({
            "matched_events": matched & event,
            "matched_hours": matched,
            "control_events": control & event,
            "control_hours": control,
        }).groupby(test.day).sum()
        me, mh, ce, ch = daily.to_numpy()[indices].sum(axis=1).T
        valid = (mh > 0) & (ch > 0)
        low, high = 100 * np.quantile(me[valid] / mh[valid] - ce[valid] / ch[valid], [0.025, 0.975])
        totals = daily.sum()
        matched_rate = totals.matched_events / totals.matched_hours
        control_rate = totals.control_events / totals.control_hours
        rows.append({
            "contrast": name,
            **totals.to_dict(),
            "matched_rate": matched_rate,
            "control_rate": control_rate,
            "difference_pp": 100 * (matched_rate - control_rate),
            "ci_low_pp": low,
            "ci_high_pp": high,
            "valid_draws": int(valid.sum()),
        })
    return pd.DataFrame(rows)


def rule_transfer(frame, models, mining):
    ridge = frame.assign(absolute_error=frame.ridge.sub(frame.actual).abs())
    chosen = mining.analyze(ridge).query("status == 'scored'")
    development = frame.year.eq(2023)
    ridge_threshold = ridge.loc[development, "absolute_error"].quantile(0.90)
    transfer, adjusted = [], []
    cluster_rule = chosen.loc[chosen.group.eq("cluster"), "antecedent"].item()
    rule_terms = dict(term.split("=") for term in cluster_rule.split(" & "))
    cluster = int(rule_terms.pop("cluster"))
    test = frame.loc[~development].copy()
    for column, value in rule_terms.items():
        test = test.loc[test[column].eq(value)]
    test["matched"] = test.cluster.eq(cluster)
    years = {
        year: (hours.index, mining.make_baskets(hours))
        for year, hours in frame.groupby("year")
    }
    for model in models:
        error = frame[model].sub(frame.actual).abs()
        model_threshold = error.loc[development].quantile(0.90)
        for definition, threshold in (
            ("model_2023_p90", model_threshold),
            ("ridge_2023_p90", ridge_threshold),
        ):
            for year, (index, baskets) in years.items():
                events = error.loc[index].gt(threshold).to_numpy()
                for rule in chosen.itertuples():
                    antecedent = frozenset(rule.antecedent.split(" & "))
                    transfer.append({"model": model, "definition": definition,
                                     "threshold": threshold, "year": year, "rule": rule.antecedent,
                                     **mining.rule_metrics(baskets, events, antecedent)})
        # Use only seasons represented in both sides of the comparison.
        test["event"] = error.loc[test.index].gt(model_threshold)
        stats = test.groupby(["season", "matched"]).event.agg(["sum", "size"]).unstack()
        stats = stats.dropna()
        target_hours = stats["size"][True]
        same_cluster = stats["sum"][True].sum() / target_hours.sum()
        other_cluster = np.average(stats["sum"][False] / stats["size"][False], weights=target_hours)
        adjusted.append({"model": model, "rule": cluster_rule,
                          "target_hours": int(target_hours.sum()),
                          "other_hours": int(stats["size"][False].sum()),
                          "target_rate": same_cluster, "season_adjusted_other_rate": other_cluster,
                          "difference_pp": 100 * (same_cluster - other_cluster)})
    return chosen, pd.DataFrame(transfer), pd.DataFrame(adjusted)


def run(files, regimes, output):
    sys.path[:0] = [str(ROOT / "team/c-gradient-boosting")]
    mining = importlib.import_module("analyze_apriori")
    frame = aligned_inputs(files, regimes)
    weights, best_2023 = add_blends(frame)
    models = [*files, "equal_blend", "global_blend", "regime_blend"]
    annual = pd.DataFrame([
        {"year": year, "phase": "calibration" if year == 2023 else "chronological_evaluation",
         "model": model, **metric(hours.actual, hours[model])}
        for year, hours in frame.groupby("year") for model in models
    ])
    chosen, transfer, adjusted = rule_transfer(frame, models, mining)
    comparisons = paired_mae(frame, best_2023)
    by_regime = pd.DataFrame([
        {"cluster": cluster, "days": hours.day.nunique(), "model": model,
         **metric(hours.actual, hours[model])}
        for cluster, hours in frame.loc[frame.year.eq(2024)].groupby("cluster")
        for model in models
    ])
    output.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output / "weights.csv")
    for name, table in {
        "annual": annual, "paired_mae": comparisons, "regime_2024": by_regime,
        "apriori_ridge": chosen, "rule_transfer": transfer, "cluster_rule_adjusted": adjusted,
        "ridge_rule_contrasts": ridge_rule_contrasts(frame),
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    inputs = {**files, "regimes": regimes, "hourly_snapshot": SNAPSHOT}
    provenance = {
        "inputs": {name: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in inputs.items()},
        "best_2023_member": best_2023, "shrinkage_prior_hours": PRIOR_HOURS,
        "block_days": BLOCK_DAYS, "draws": DRAWS, "seed": SEED,
        "scope": "2023 calibration and exploratory 2024 chronological evaluation",
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return annual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ["ridge", *MEMBERS]:
        parser.add_argument("--" + model.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--regimes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/team/postprocess")
    args = parser.parse_args()
    annual = run({model: getattr(args, model) for model in ["ridge", *MEMBERS]}, args.regimes, args.output)
    columns = ["model", "mae", "rmse", "rate_5pct"]
    print(annual.loc[annual.year.eq(2024), columns].to_string(index=False))


if __name__ == "__main__":
    main()
