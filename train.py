from __future__ import annotations

import argparse

from src.energy_efficiency import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate energy-efficiency regression models.")
    parser.add_argument("--data", default="ENB2012_data.xlsx", help="Path to the ENB2012 Excel dataset.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for trained model artifacts.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for metric and analysis reports.")
    parser.add_argument("--tune-rf", action="store_true", help="Run RandomizedSearchCV for Random Forest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        data_path=args.data,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        tune_rf=args.tune_rf,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
