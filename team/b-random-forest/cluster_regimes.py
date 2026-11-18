"""Fit DK1 market regimes on earlier features and export daily cluster labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HOURS = range(24)
LAGGED_PRICE = [f"DK1_lag1_h{hour}" for hour in HOURS]
FORECAST_WEATHER = {
    "forecast_temperature_mean": "temperature_2m",
    "forecast_wind_speed_mean": "wind_speed_10m",
    "forecast_cloud_cover_mean": "cloud_cover",
}
WEATHER_COLUMNS = {
    name: [f"weather_DK1_{source}_h{hour}" for hour in HOURS]
    for name, source in FORECAST_WEATHER.items()
}


def load_descriptors(path):
    """Represent a day using lagged price level and shape and forecast weather."""
    columns = ["day", *LAGGED_PRICE]
    for names in WEATHER_COLUMNS.values():
        columns.extend(names)
    features = pd.read_csv(path, usecols=columns, parse_dates=["day"])
    days = pd.DatetimeIndex(features.pop("day"), name="day")
    if days.hasnans or days.has_duplicates or not days.equals(days.normalize()):
        raise ValueError("Expected unique, non-missing local calendar days in features")
    features.index = days
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Market-regime inputs must be finite")
    price = features[LAGGED_PRICE]
    descriptors = pd.DataFrame(
        {
            "prior_price_mean": price.mean(axis=1),
            "prior_price_sd": price.std(axis=1, ddof=0),
            "prior_morning_evening_spread": (
                price.iloc[:, 17:22].mean(axis=1) - price.iloc[:, 6:11].mean(axis=1)
            ),
            **{
                name: features[names].mean(axis=1)
                for name, names in WEATHER_COLUMNS.items()
            },
        },
        index=days,
    )
    return descriptors.sort_index()


def run(features_path, output_dir, clusters):
    descriptors = load_descriptors(features_path)
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")
    train = descriptors.loc[descriptors.index < start]
    evaluation = descriptors.loc[(descriptors.index >= start) & (descriptors.index < end)]
    expected = pd.date_range(start, end, inclusive="left", freq="D", name="day")
    if not evaluation.index.equals(expected):
        raise ValueError("Expected exactly one feature row per day in 2023-2024")
    if not 2 <= clusters < len(train):
        raise ValueError("Need 2 or more clusters and more training days than clusters")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    kmeans = KMeans(n_clusters=clusters, n_init=20, random_state=42)
    train_labels = kmeans.fit_predict(train_scaled)
    if len(np.unique(train_labels)) != clusters:
        raise ValueError("Fewer distinct clusters than requested; inspect the inputs")
    labels = kmeans.predict(scaler.transform(evaluation))
    counts = {"train_days": np.bincount(train_labels, minlength=clusters)}
    for year in (2023, 2024):
        counts[f"{year}_days"] = np.bincount(
            labels[evaluation.index.year == year], minlength=clusters
        )
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=descriptors.columns
    )
    centers = pd.concat([pd.DataFrame(counts), centers], axis=1).rename_axis("cluster")
    dbi = davies_bouldin_score(train_scaled, train_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(labels, index=evaluation.index, name="cluster").to_csv(
        output_dir / "regimes.csv", date_format="%Y-%m-%d"
    )
    centers.to_csv(output_dir / "regime_centers.csv")
    pd.DataFrame(
        [{"clusters": clusters, "training_days": len(train), "training_davies_bouldin": dbi}]
    ).to_csv(output_dir / "regime_diagnostics.csv", index=False)
    print(f"k={clusters}: training Davies-Bouldin index {dbi:.3f}")
    print(f"Wrote {len(evaluation)} daily labels to {output_dir / 'regimes.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.features, args.output_dir, args.clusters)


if __name__ == "__main__":
    main()
