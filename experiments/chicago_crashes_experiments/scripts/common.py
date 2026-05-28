from __future__ import annotations

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit

BASE_URL = "https://data.cityofchicago.org/resource/85ca-t3if.csv"
COMMUNITY_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"

DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2023-12-31"


def fetch_crash_data(
    start_date: str,
    end_date: str,
    limit: int = 200000,
    timeout: int = 120,
) -> pd.DataFrame:
    params = {
        "$select": ",".join(
            [
                "crash_date",
                "latitude",
                "longitude",
                "injuries_total",
                "crash_hour",
                "weather_condition",
                "lighting_condition",
                "first_crash_type",
                "trafficway_type",
                "posted_speed_limit",
                "prim_contributory_cause",
            ]
        ),
        "$where": (
            f"crash_date between '{start_date}T00:00:00' and '{end_date}T23:59:59' "
            "AND latitude IS NOT NULL AND longitude IS NOT NULL"
        ),
        "$limit": limit,
    }

    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def clean_crash_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df.dropna(subset=["crash_date", "latitude", "longitude"]).copy()

    num_cols = [
        "latitude",
        "longitude",
        "injuries_total",
        "posted_speed_limit",
        "crash_hour",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_cols = [
        "weather_condition",
        "lighting_condition",
        "first_crash_type",
        "trafficway_type",
        "prim_contributory_cause",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    df["date"] = df["crash_date"].dt.floor("D")
    df["injury_flag"] = (df["injuries_total"].fillna(0) > 0).astype(int)
    return df


def load_community_areas() -> gpd.GeoDataFrame:
    areas = gpd.read_file(COMMUNITY_URL)
    areas.columns = [col.lower() for col in areas.columns]

    if set(areas.columns) == {"geometry"}:
        geo = requests.get(COMMUNITY_URL, timeout=60).json()
        areas = gpd.GeoDataFrame.from_features(geo["features"], crs="EPSG:4326")
        areas.columns = [col.lower() for col in areas.columns]

    areas = areas.to_crs("EPSG:4326")

    id_col_candidates = ["area_numbe", "area_number", "area_num", "area_num_1"]
    name_col_candidates = [
        "community",
        "community_area",
        "community_name",
        "area_name",
        "comarea",
        "comarea_name",
        "name",
    ]

    id_col = next((col for col in id_col_candidates if col in areas.columns), None)
    name_col = next((col for col in name_col_candidates if col in areas.columns), None)

    if id_col is None or name_col is None:
        raise ValueError(f"Missing expected columns in community areas: {areas.columns}")

    areas = areas.rename(columns={id_col: "community_area_id", name_col: "community_area"})
    areas["community_area_id"] = pd.to_numeric(
        areas["community_area_id"], errors="coerce"
    ).astype("Int64")

    areas = areas[["community_area_id", "community_area", "geometry"]].dropna(
        subset=["community_area_id"]
    )
    return areas


def assign_community_areas(
    df: pd.DataFrame, areas: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return gpd.sjoin(gdf, areas, how="left", predicate="within")


def build_daily_area_panel(
    gdf: gpd.GeoDataFrame, areas: gpd.GeoDataFrame
) -> pd.DataFrame:
    daily_area = (
        gdf.groupby(["date", "community_area_id", "community_area"])
        .size()
        .rename("n_crashes")
        .reset_index()
    )

    all_dates = pd.date_range(daily_area["date"].min(), daily_area["date"].max(), freq="D")
    all_areas = areas[["community_area_id", "community_area"]].drop_duplicates()

    grid = pd.MultiIndex.from_product(
        [all_dates, all_areas["community_area_id"]],
        names=["date", "community_area_id"],
    ).to_frame(index=False)
    grid = grid.merge(all_areas, on="community_area_id", how="left")

    daily_area = grid.merge(
        daily_area,
        on=["date", "community_area_id", "community_area"],
        how="left",
    )
    daily_area["n_crashes"] = daily_area["n_crashes"].fillna(0).astype(int)
    return daily_area


def add_time_features(daily_area: pd.DataFrame) -> pd.DataFrame:
    daily_area = daily_area.sort_values(["community_area_id", "date"]).reset_index(
        drop=True
    )
    daily_area["dow"] = daily_area["date"].dt.dayofweek
    daily_area["month"] = daily_area["date"].dt.month
    daily_area["weekofyear"] = daily_area["date"].dt.isocalendar().week.astype(int)
    daily_area["is_weekend"] = (daily_area["dow"] >= 5).astype(int)
    daily_area["dow_sin"] = np.sin(2 * np.pi * daily_area["dow"] / 7)
    daily_area["dow_cos"] = np.cos(2 * np.pi * daily_area["dow"] / 7)
    return daily_area


def add_lag_features(
    daily_area: pd.DataFrame, lags: tuple[int, ...] = (1, 7, 14), rolling: int = 7
) -> pd.DataFrame:
    daily_area = daily_area.sort_values(["community_area_id", "date"]).reset_index(
        drop=True
    )
    for lag in lags:
        daily_area[f"lag_{lag}"] = (
            daily_area.groupby("community_area_id")["n_crashes"].shift(lag)
        )

    daily_area[f"rolling_{rolling}"] = (
        daily_area.groupby("community_area_id")["n_crashes"]
        .shift(1)
        .rolling(rolling)
        .mean()
        .reset_index(level=0, drop=True)
    )

    lag_cols = [f"lag_{lag}" for lag in lags] + [f"rolling_{rolling}"]
    daily_area[lag_cols] = daily_area[lag_cols].fillna(0)
    return daily_area


def ensure_features(daily_area: pd.DataFrame) -> pd.DataFrame:
    if "dow" not in daily_area.columns:
        daily_area = add_time_features(daily_area)
    if "lag_1" not in daily_area.columns:
        daily_area = add_lag_features(daily_area)
    return daily_area


def time_based_split(daily_area: pd.DataFrame, test_frac: float = 0.2):
    dates = np.sort(daily_area["date"].unique())
    cutoff = int(len(dates) * (1 - test_frac))
    train_dates = dates[:cutoff]
    test_dates = dates[cutoff:]
    train_mask = daily_area["date"].isin(train_dates)
    test_mask = daily_area["date"].isin(test_dates)
    return train_mask, test_mask


def spatiotemporal_folds(
    data: pd.DataFrame,
    n_time_splits: int = 4,
    test_area_frac: float = 0.2,
    random_state: int = 42,
):
    dates = np.sort(data["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_time_splits)
    folds = []

    for fold_idx, (train_date_idx, test_date_idx) in enumerate(tscv.split(dates), start=1):
        train_dates = dates[train_date_idx]
        test_dates = dates[test_date_idx]

        train_time_mask = data["date"].isin(train_dates)
        test_time_mask = data["date"].isin(test_dates)

        train_subset = data.loc[train_time_mask]
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=test_area_frac,
            random_state=random_state + fold_idx,
        )
        _, test_area_idx = next(
            gss.split(train_subset, groups=train_subset["community_area_id"])
        )
        test_area_ids = train_subset.iloc[test_area_idx]["community_area_id"].unique()

        train_area_mask = ~data["community_area_id"].isin(test_area_ids)
        test_area_mask = data["community_area_id"].isin(test_area_ids)

        train_mask = train_time_mask & train_area_mask
        test_mask = test_time_mask & test_area_mask

        folds.append(
            {
                "fold": fold_idx,
                "train_idx": data.index[train_mask].to_numpy(),
                "test_idx": data.index[test_mask].to_numpy(),
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_area_ids": data.loc[train_mask, "community_area_id"].unique(),
                "test_area_ids": test_area_ids,
            }
        )

    return folds


def append_metrics(metrics_path: Path, row: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(metrics_path, index=False)
