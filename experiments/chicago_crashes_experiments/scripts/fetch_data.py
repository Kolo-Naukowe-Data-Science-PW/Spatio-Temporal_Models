from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    add_lag_features,
    add_time_features,
    assign_community_areas,
    build_daily_area_panel,
    clean_crash_data,
    fetch_crash_data,
    load_community_areas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and prepare Chicago crash data.")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=200000)
    parser.add_argument(
        "--raw-output",
        default=str(Path("data/raw/crashes_raw.csv")),
        help="Raw CSV output path.",
    )
    parser.add_argument(
        "--processed-output",
        default=str(Path("data/processed/daily_area.csv")),
        help="Processed panel output path.",
    )
    parser.add_argument(
        "--areas-output",
        default=str(Path("data/processed/community_areas.geojson")),
        help="Community area GeoJSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_output)
    processed_path = Path(args.processed_output)
    areas_path = Path(args.areas_output)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    areas_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_crash_data(args.start, args.end, limit=args.limit)
    df.to_csv(raw_path, index=False)

    df = clean_crash_data(df)
    areas = load_community_areas()
    areas.to_file(areas_path, driver="GeoJSON")

    gdf = assign_community_areas(df, areas)
    daily_area = build_daily_area_panel(gdf, areas)
    daily_area = add_time_features(daily_area)
    daily_area = add_lag_features(daily_area)

    daily_area.to_csv(processed_path, index=False)

    print("Saved:")
    print(f"- {raw_path}")
    print(f"- {processed_path}")
    print(f"- {areas_path}")


if __name__ == "__main__":
    main()
