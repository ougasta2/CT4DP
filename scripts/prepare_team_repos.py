"""Assemble new local A–D Git repositories from tracked starter sources."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from run_team_benchmark import ROOT, prepare

STARTERS = ("a-lasso", "b-random-forest", "c-gradient-boosting", "d-mlp")
PACKAGES = {"joblib", "numpy", "pandas", "python-dateutil", "pytz", "scikit-learn",
            "scipy", "six", "threadpoolctl", "tzdata"}


def write_snapshot(data):
    features, targets, hourly = prepare()
    features = features.loc[features.index < "2025-01-01"]
    targets = targets.loc[features.index]
    frames = {
        "features": features.rename_axis("day"),
        "targets": targets.rename_axis("day"),
        "hourly": hourly.loc[hourly.day.isin(features.index), ["day", "hour", "DK1"]]
                        .rename_axis("HourUTC"),
    }
    data.mkdir()
    hashes = {}
    for name, frame in frames.items():
        path = data / f"{name}.csv.gz"
        frame.to_csv(path, compression={"method": "gzip", "mtime": 0})
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {
        "sha256": hashes,
        "feature_days": len(features),
        "first_day": str(features.index.min().date()),
        "last_day": str(features.index.max().date()),
        "evaluation": {"start": "2023-01-01", "end_exclusive": "2025-01-01",
                       "status": "historical development"},
        "cutoff": "intended 10:00 Europe/Copenhagen on D-1",
        "sources": "Same five-zone lagged price, calendar and archived JMA weather inputs as the main benchmark.",
    }
    (data / "manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def ridge_summary(metrics):
    with metrics.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [(row["year"], row["hours"]) for row in rows] != [("2023", "8760"), ("2024", "8784")]:
        raise ValueError("Run the full 2023–2024 Ridge benchmark before creating the repositories.")
    lines = [
        "# Frozen Ridge baseline",
        "",
        "Training-only scaling and multi-output Ridge with alpha=100, refit monthly.",
        "",
        "| Year | Hours | MAE (EUR/MWh) | ±5% hits / eligible hours |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f'| {row["year"]} | {int(row["hours"]):,} | {float(row["mae"]):.2f} | '
                     f'{int(row["hits_5pct"]):,} / {int(row["eligible"]):,} ({float(row["rate_5pct"]):.2%}) |')
    lines += ["", "Both years are historical development. See [shared protocol](BENCHMARK.md)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "team_repos")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error(f"Destination already exists: {destination}. Use a new directory.")
    source = ROOT / "team_starters"
    ridge_metrics = ROOT / "results/team/ridge/metrics.csv"
    if not ridge_metrics.is_file() or any(not (source / name / "model.py").is_file() for name in STARTERS):
        parser.error("Run the full Ridge benchmark and ensure all four model starters are present.")
    summary = ridge_summary(ridge_metrics)
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    requirements = "\n".join(line for line in lock if line.split("==", 1)[0] in PACKAGES) + "\n"
    destination.mkdir(parents=True)
    snapshot = destination / "_benchmark_data"
    write_snapshot(snapshot)
    for name in STARTERS:
        repo = destination / name
        shutil.copytree(source / name, repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(snapshot, repo / "data")
        shutil.copy2(source / "benchmark.py", repo)
        shutil.copy2(source / "BENCHMARK.md", repo)
        shutil.copy2(source / "DATA_LICENSE.md", repo)
        shutil.copy2(ROOT / "scripts/team_benchmark_core.py", repo)
        (repo / "RIDGE_BASELINE.md").write_text(summary, encoding="utf-8")
        if name == "c-gradient-boosting":
            shutil.copy2(ROOT / "src/energy_pilot/mining.py", repo)
        (repo / "requirements.txt").write_text(requirements, encoding="utf-8")
        (repo / ".gitignore").write_text(
            "data/*\n!data/features.csv.gz\n!data/targets.csv.gz\n!data/hourly.csv.gz\n"
            "!data/manifest.json\nresults/\n.venv/\n__pycache__/\n*.pyc\n", encoding="utf-8",
        )
        subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True)


if __name__ == "__main__":
    main()
