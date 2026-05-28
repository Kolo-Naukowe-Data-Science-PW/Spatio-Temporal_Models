from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all models with shared settings.")
    parser.add_argument("--data", default=str(Path("data/processed/daily_area.csv")))
    parser.add_argument("--areas", default=str(Path("data/processed/community_areas.geojson")))
    parser.add_argument("--metrics-output", default=str(Path("outputs/metrics.csv")))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-area-frac", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--georegression-models", default="strf,stst,gwr")
    parser.add_argument("--kriging-max-train-samples", type=int, default=20000)
    parser.add_argument("--graph-epochs", type=int, default=30)
    parser.add_argument("--graph-hidden-dim", type=int, default=32)
    parser.add_argument("--graph-learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def run_script(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    args = parse_args()
    shared = [
        "--data",
        args.data,
        "--metrics-output",
        args.metrics_output,
        "--cv-splits",
        str(args.cv_splits),
        "--test-area-frac",
        str(args.test_area_frac),
        "--random-state",
        str(args.random_state),
    ]

    run_script(["scripts/train_xgboost.py", *shared])

    run_script(
        [
            "scripts/train_kriging.py",
            *shared,
            "--areas",
            args.areas,
            "--max-train-samples",
            str(args.kriging_max_train_samples),
        ]
    )

    run_script(["scripts/train_poisson.py", *shared])

    run_script(
        [
            "scripts/train_graph_model.py",
            *shared,
            "--areas",
            args.areas,
            "--epochs",
            str(args.graph_epochs),
            "--hidden-dim",
            str(args.graph_hidden_dim),
            "--learning-rate",
            str(args.graph_learning_rate),
        ]
    )

    run_script(
        [
            "scripts/train_georegression.py",
            *shared,
            "--areas",
            args.areas,
            "--models",
            args.georegression_models,
        ]
    )


if __name__ == "__main__":
    main()
