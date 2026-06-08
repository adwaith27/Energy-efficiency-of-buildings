from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, Info, start_http_server

from src.energy_efficiency import load_dataset, train_and_evaluate


MODEL_R2 = Gauge("energy_model_r2", "R2 score by model and target.", ["model", "target"])
MODEL_RMSE = Gauge("energy_model_rmse", "RMSE by model and target.", ["model", "target"])
MODEL_MAE = Gauge("energy_model_mae", "MAE by model and target.", ["model", "target"])
FEATURE_IMPORTANCE = Gauge("energy_feature_importance", "Feature importance from ExtraTreesRegressor.", ["feature"])
DATASET_ROWS = Gauge("energy_dataset_rows", "Number of rows in the dataset.")
DATASET_COLUMNS = Gauge("energy_dataset_columns", "Number of columns in the dataset.")
TARGET_MEAN = Gauge("energy_target_mean", "Mean target value.", ["target"])
TARGET_STD = Gauge("energy_target_std", "Target standard deviation.", ["target"])
RUN_INFO = Info("energy_efficiency_run", "Current energy-efficiency monitoring run.")


def publish_reports(reports_dir: Path, data_path: Path) -> None:
    metrics = pd.read_csv(reports_dir / "model_metrics.csv")
    for row in metrics.itertuples(index=False):
        MODEL_R2.labels(model=row.model, target=row.target).set(row.r2)
        MODEL_RMSE.labels(model=row.model, target=row.target).set(row.rmse)
        MODEL_MAE.labels(model=row.model, target=row.target).set(row.mae)

    importance = pd.read_csv(reports_dir / "feature_importance.csv")
    for row in importance.itertuples(index=False):
        FEATURE_IMPORTANCE.labels(feature=row.feature).set(row.importance)

    df = load_dataset(data_path)
    DATASET_ROWS.set(len(df))
    DATASET_COLUMNS.set(len(df.columns))
    for target in ["heating_load", "cooling_load"]:
        TARGET_MEAN.labels(target=target).set(df[target].mean())
        TARGET_STD.labels(target=target).set(df[target].std())

    RUN_INFO.info({"data_path": str(data_path), "reports_dir": str(reports_dir)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose project metrics for Prometheus and Grafana.")
    parser.add_argument("--data", default="ENB2012_data.xlsx")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--refresh-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    reports_dir = Path(args.reports_dir)
    if not (reports_dir / "model_metrics.csv").exists():
        train_and_evaluate(data_path=data_path, reports_dir=reports_dir)

    start_http_server(args.port)
    while True:
        publish_reports(reports_dir, data_path)
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()
