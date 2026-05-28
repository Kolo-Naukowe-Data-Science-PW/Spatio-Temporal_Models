from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pykrige.ok import OrdinaryKriging

from common import append_metrics, spatiotemporal_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial-only kriging baseline.")
    parser.add_argument(
        "--data",
        default=str(Path("data/processed/daily_area.csv")),
    )
    parser.add_argument(
        "--areas",
        default=str(Path("data/processed/community_areas.geojson")),
    )
    parser.add_argument(
        "--metrics-output",
        default=str(Path("outputs/metrics.csv")),
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-area-frac", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])

    areas_gdf = gpd.read_file(args.areas)
    areas_proj = areas_gdf.to_crs("EPSG:26916")
    areas_proj["centroid"] = areas_proj.geometry.centroid
    areas_centroids = areas_proj.set_geometry("centroid").to_crs("EPSG:4326")
    areas_centroids["lon"] = areas_centroids.geometry.x
    areas_centroids["lat"] = areas_centroids.geometry.y

    df = df.merge(
        areas_centroids[["community_area_id", "lon", "lat"]],
        on="community_area_id",
        how="left",
    )

    folds = spatiotemporal_folds(
        df,
        n_time_splits=args.cv_splits,
        test_area_frac=args.test_area_frac,
        random_state=args.random_state,
    )

    rows = []
    for fold in folds:
        train_df = df.loc[fold["train_idx"]].copy()
        test_df = df.loc[fold["test_idx"]].copy()

        if len(train_df) > args.max_train_samples:
            train_df = train_df.sample(args.max_train_samples, random_state=args.random_state)

        ok = OrdinaryKriging(
            train_df["lon"].to_numpy(),
            train_df["lat"].to_numpy(),
            train_df["n_crashes"].to_numpy(),
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )

        preds, _ = ok.execute(
            "points",
            test_df["lon"].to_numpy(),
            test_df["lat"].to_numpy(),
        )

        mae = mean_absolute_error(test_df["n_crashes"], preds)
        rmse = np.sqrt(mean_squared_error(test_df["n_crashes"], preds))
        row = {
            "model": "kriging_spatial_only",
            "split": "spatiotemporal_cv",
            "fold": fold["fold"],
            "mae": mae,
            "rmse": rmse,
            "cv_splits": args.cv_splits,
            "test_area_frac": args.test_area_frac,
        }
        append_metrics(Path(args.metrics_output), row)
        rows.append(row)

    mean_mae = float(np.mean([row["mae"] for row in rows]))
    mean_rmse = float(np.mean([row["rmse"] for row in rows]))
    append_metrics(
        Path(args.metrics_output),
        {
            "model": "kriging_spatial_only",
            "split": "spatiotemporal_cv_mean",
            "fold": "mean",
            "mae": mean_mae,
            "rmse": mean_rmse,
            "cv_splits": args.cv_splits,
            "test_area_frac": args.test_area_frac,
        },
    )

    print({"mae": mean_mae, "rmse": mean_rmse})


if __name__ == "__main__":
    main()
