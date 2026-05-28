from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from common import append_metrics, spatiotemporal_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph temporal model baseline.")
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
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--test-area-frac", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def build_adjacency(areas: gpd.GeoDataFrame) -> np.ndarray:
    areas = areas.to_crs("EPSG:4326")
    geometries = areas["geometry"].tolist()
    n = len(geometries)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if geometries[i].touches(geometries[j]):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def build_dataset(
    df: pd.DataFrame,
    areas: gpd.GeoDataFrame,
    area_ids: np.ndarray,
    date_range: np.ndarray,
) -> StaticGraphTemporalSignal:
    areas = areas[areas["community_area_id"].isin(area_ids)].copy()
    areas = areas.sort_values("community_area_id").reset_index(drop=True)

    df = df[df["community_area_id"].isin(area_ids)].copy()
    pivot = df.pivot_table(
        index="date",
        columns="community_area_id",
        values="n_crashes",
        fill_value=0,
    )
    pivot = pivot.reindex(date_range)
    pivot = pivot.reindex(columns=areas["community_area_id"], fill_value=0)

    values = pivot.to_numpy().astype(np.float32)
    features = [values[t][:, None] for t in range(values.shape[0] - 1)]
    targets = [values[t + 1][:, None] for t in range(values.shape[0] - 1)]

    adj = build_adjacency(areas)
    edge_index = np.vstack(np.nonzero(adj))
    edge_weight = adj[edge_index[0], edge_index[1]]

    return StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)


class GConvGRUModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.recurrent = GConvGRU(in_channels, hidden_dim, K=2)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        return self.linear(h)


def evaluate(model, dataset, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for snapshot in dataset:
            y_hat = model(
                snapshot.x.to(device),
                snapshot.edge_index.to(device),
                snapshot.edge_attr.to(device),
            )
            preds.append(y_hat.cpu().numpy())
            trues.append(snapshot.y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    return mae, rmse


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])
    areas = gpd.read_file(args.areas)

    folds = spatiotemporal_folds(
        df,
        n_time_splits=args.cv_splits,
        test_area_frac=args.test_area_frac,
        random_state=args.random_state,
    )

    rows = []
    for fold in folds:
        train_dataset = build_dataset(
            df,
            areas,
            fold["train_area_ids"],
            fold["train_dates"],
        )
        test_dataset = build_dataset(
            df,
            areas,
            fold["test_area_ids"],
            fold["test_dates"],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GConvGRUModel(in_channels=1, hidden_dim=args.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.MSELoss()

        for _ in range(args.epochs):
            model.train()
            for snapshot in train_dataset:
                optimizer.zero_grad()
                y_hat = model(
                    snapshot.x.to(device),
                    snapshot.edge_index.to(device),
                    snapshot.edge_attr.to(device),
                )
                loss = loss_fn(y_hat, snapshot.y.to(device))
                loss.backward()
                optimizer.step()

        mae, rmse = evaluate(model, test_dataset, device)
        row = {
            "model": "gconvgru",
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
            "model": "gconvgru",
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
