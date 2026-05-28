from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from georegression.weight_model import WeightModel
from georegression.stacking_model import StackingWeightModel

from common import append_metrics, ensure_features, spatiotemporal_folds


FEATURE_COLS = [
    "dow",
    "month",
    "weekofyear",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoRegression models.")
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
    parser.add_argument("--neighbour-count", type=float, default=0.3)
    parser.add_argument("--kernel-type", default="gaussian")
    parser.add_argument(
        "--models",
        default="strf,stst,gwr",
        help="Comma-separated: strf, stst, gwr",
    )
    return parser.parse_args()


def build_spatiotemporal_arrays(df: pd.DataFrame, areas_path: str):
    areas = gpd.read_file(areas_path)
    areas_proj = areas.to_crs("EPSG:26916")
    areas_proj["centroid"] = areas_proj.geometry.centroid
    areas_centroids = areas_proj.set_geometry("centroid").to_crs("EPSG:4326")
    areas_centroids["lon"] = areas_centroids.geometry.x
    areas_centroids["lat"] = areas_centroids.geometry.y

    df = df.merge(
        areas_centroids[["community_area_id", "lon", "lat"]],
        on="community_area_id",
        how="left",
    )
    if df[["lon", "lat"]].isna().any().any():
        raise ValueError("Missing centroids for some community areas.")

    X = df[FEATURE_COLS].to_numpy()
    points = df[["lon", "lat"]].to_numpy()

    t0 = df["date"].min()
    times = (df["date"] - t0).dt.days.to_numpy().reshape(-1, 1)

    X_plus = np.concatenate([X, points, times], axis=1)
    coords = np.concatenate([points, times], axis=1)
    return X_plus, coords


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"model": model_name, "mae": mae, "rmse": rmse}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = ensure_features(df)

    X_plus, coords = build_spatiotemporal_arrays(df, args.areas)
    y = df["n_crashes"].to_numpy()

    models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    metrics_path = Path(args.metrics_output)

    folds = spatiotemporal_folds(
        df,
        n_time_splits=args.cv_splits,
        test_area_frac=args.test_area_frac,
        random_state=args.random_state,
    )

    def append_mean(model_name: str, rows: list[dict]) -> None:
        mean_mae = float(np.mean([row["mae"] for row in rows]))
        mean_rmse = float(np.mean([row["rmse"] for row in rows]))
        append_metrics(
            metrics_path,
            {
                "model": model_name,
                "split": "spatiotemporal_cv_mean",
                "fold": "mean",
                "mae": mean_mae,
                "rmse": mean_rmse,
                "cv_splits": args.cv_splits,
                "test_area_frac": args.test_area_frac,
            },
        )

    if "strf" in models:
        rows = []
        for fold in folds:
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]

            X_train = X_plus[train_idx]
            y_train = y[train_idx]
            coords_train = coords[train_idx]

            X_test = X_plus[test_idx]
            y_test = y[test_idx]
            coords_test = coords[test_idx]

            model = WeightModel(
                RandomForestRegressor(n_estimators=200, random_state=args.random_state),
                distance_measure="euclidean",
                kernel_type=args.kernel_type,
                neighbour_count=args.neighbour_count,
                cache_data=True,
            )
            model.fit(X_train, y_train, [coords_train])
            preds = model.predict_by_fit(
                X_test,
                [coords_test],
            )
            metrics = evaluate_model("strf", y_test, preds)
            metrics.update(
                {
                    "split": "spatiotemporal_cv",
                    "fold": fold["fold"],
                    "cv_splits": args.cv_splits,
                    "test_area_frac": args.test_area_frac,
                }
            )
            append_metrics(metrics_path, metrics)
            rows.append(metrics)
        append_mean("strf", rows)

    if "stst" in models:
        rows = []
        for fold in folds:
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]

            X_train = X_plus[train_idx]
            y_train = y[train_idx]
            coords_train = coords[train_idx]

            X_test = X_plus[test_idx]
            y_test = y[test_idx]
            coords_test = coords[test_idx]

            model = StackingWeightModel(
                DecisionTreeRegressor(splitter="random", max_depth=X_train.shape[1]),
                distance_measure="euclidean",
                kernel_type=args.kernel_type,
                neighbour_count=args.neighbour_count,
                neighbour_leave_out_rate=0.1,
                cache_data=True,
            )
            model.fit(X_train, y_train, [coords_train])
            preds = model.predict_by_fit(
                X_test,
                [coords_test],
            )
            metrics = evaluate_model("stst", y_test, preds)
            metrics.update(
                {
                    "split": "spatiotemporal_cv",
                    "fold": fold["fold"],
                    "cv_splits": args.cv_splits,
                    "test_area_frac": args.test_area_frac,
                }
            )
            append_metrics(metrics_path, metrics)
            rows.append(metrics)
        append_mean("stst", rows)

    if "gwr" in models:
        rows = []
        for fold in folds:
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]

            X_train = X_plus[train_idx]
            y_train = y[train_idx]
            coords_train = coords[train_idx]

            X_test = X_plus[test_idx]
            y_test = y[test_idx]
            coords_test = coords[test_idx]

            model = WeightModel(
                LinearRegression(),
                distance_measure="euclidean",
                kernel_type=args.kernel_type,
                neighbour_count=args.neighbour_count,
                cache_data=True,
            )
            model.fit(X_train, y_train, [coords_train])
            preds = model.predict_by_fit(
                X_test,
                [coords_test],
            )
            metrics = evaluate_model("gwr", y_test, preds)
            metrics.update(
                {
                    "split": "spatiotemporal_cv",
                    "fold": fold["fold"],
                    "cv_splits": args.cv_splits,
                    "test_area_frac": args.test_area_frac,
                }
            )
            append_metrics(metrics_path, metrics)
            rows.append(metrics)
        append_mean("gwr", rows)


if __name__ == "__main__":
    main()
