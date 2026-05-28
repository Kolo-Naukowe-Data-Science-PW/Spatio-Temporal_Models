from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common import append_metrics, ensure_features, spatiotemporal_folds


FEATURE_COLS = [
    "community_area_id",
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
    parser = argparse.ArgumentParser(description="Train Poisson regression baseline.")
    parser.add_argument(
        "--data",
        default=str(Path("data/processed/daily_area.csv")),
    )
    parser.add_argument(
        "--metrics-output",
        default=str(Path("outputs/metrics.csv")),
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-area-frac", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = ensure_features(df)

    X = df[FEATURE_COLS].copy()
    y = df["n_crashes"].values

    X = pd.get_dummies(X, columns=["community_area_id"], drop_first=False)
    folds = spatiotemporal_folds(
        df,
        n_time_splits=args.cv_splits,
        test_area_frac=args.test_area_frac,
        random_state=args.random_state,
    )

    metrics_rows = []
    for fold in folds:
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        X_train = X.loc[train_idx]
        y_train = y[train_idx]
        X_test = X.loc[test_idx]
        y_test = y[test_idx]

        model = PoissonRegressor(
            alpha=args.alpha,
            max_iter=args.max_iter,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        row = {
            "model": "poisson",
            "split": "spatiotemporal_cv",
            "fold": fold["fold"],
            "mae": mae,
            "rmse": rmse,
            "cv_splits": args.cv_splits,
            "test_area_frac": args.test_area_frac,
        }
        append_metrics(Path(args.metrics_output), row)
        metrics_rows.append(row)

    mean_mae = float(np.mean([row["mae"] for row in metrics_rows]))
    mean_rmse = float(np.mean([row["rmse"] for row in metrics_rows]))
    append_metrics(
        Path(args.metrics_output),
        {
            "model": "poisson",
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
