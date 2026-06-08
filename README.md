# Energy Efficiency Regression and Monitoring

Machine learning project for predicting building heating load and cooling load from the ENB2012 energy-efficiency dataset. The repository includes a reproducible training pipeline, generated model reports, a Prometheus metrics exporter, and a provisioned Grafana dashboard for monitoring model quality.

## Overview

The project trains and compares multiple regression models on building design features such as compactness, surface area, roof area, wall area, height, orientation, and glazing. The two prediction targets are:

- Heating load
- Cooling load

The monitoring stack exposes the training results as Prometheus metrics and visualizes them in Grafana.

## Repository Contents

```text
.
+-- ENB2012_data.xlsx                         # Included dataset
+-- dataset_explanation_paper.pdf             # Dataset reference paper
+-- Building_Energy_Efficiency.ipynb          # Original exploratory notebook
+-- src/energy_efficiency.py                  # Data loading, training, evaluation logic
+-- train.py                                  # CLI entry point for model training
+-- metrics_server.py                         # Prometheus metrics exporter
+-- Dockerfile                                # Metrics exporter container
+-- docker-compose.yml                        # Grafana + Prometheus + exporter stack
+-- monitoring/
|   +-- prometheus/prometheus.yml
|   +-- grafana/
|       +-- dashboards/energy-efficiency.json
|       +-- provisioning/
+-- pyproject.toml                            # Python dependencies
+-- uv.lock                                   # Locked dependency versions
+-- Makefile                                  # Convenience commands
```

## Dataset

The dataset is included in this repository as:

```text
ENB2012_data.xlsx
```

It contains 768 simulated building designs with 8 input features and 2 target variables.

Original columns are renamed by the training code:

| Original | Project column | Description |
| --- | --- | --- |
| X1 | `relative_compactness` | Building compactness |
| X2 | `surface_area` | Total surface area |
| X3 | `wall_area` | Wall area |
| X4 | `roof_area` | Roof area |
| X5 | `overall_height` | Building height |
| X6 | `orientation` | Building orientation |
| X7 | `glazing_area` | Window/glazing area |
| X8 | `glazing_area_distribution` | Glazing distribution |
| Y1 | `heating_load` | Heating load target |
| Y2 | `cooling_load` | Cooling load target |

The dataset explanation paper is also included:

```text
dataset_explanation_paper.pdf
```

## Models

The training pipeline compares:

- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Random Forest Regression

The default training run also performs cross-validation based tuning for Ridge and Lasso alpha values. Random Forest uses a strong default configuration unless the optional `--tune-rf` flag is used.

## Requirements

Install these before running the project:

- Python 3.10 or newer
- `uv` Python package manager
- Docker Desktop or Docker Engine
- Docker Compose

Check versions:

```bash
python --version
uv --version
docker --version
docker compose version
```

## Fresh Clone Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd Energy-Efficiency
```

Install Python dependencies:

```bash
uv sync
```

Optional notebook dependencies:

```bash
uv sync --extra notebook
```

## Train the Models

Run the standard training pipeline:

```bash
uv run python train.py
```

This creates:

```text
artifacts/
+-- lasso.joblib
+-- polynomial_regression.joblib
+-- random_forest.joblib
+-- ridge.joblib

reports/
+-- correlations.csv
+-- feature_importance.csv
+-- model_metrics.csv
+-- tuned_parameters.json
```

The main model report is `reports/model_metrics.csv`.

Example metrics from a default run:

| Model | Target | R2 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Ridge | Heating load | 0.912 | 3.025 | 2.182 |
| Ridge | Cooling load | 0.893 | 3.145 | 2.195 |
| Lasso | Heating load | 0.912 | 3.026 | 2.183 |
| Lasso | Cooling load | 0.893 | 3.146 | 2.195 |
| Polynomial Regression | Heating load | 0.994 | 0.803 | 0.604 |
| Polynomial Regression | Cooling load | 0.968 | 1.726 | 1.191 |
| Random Forest | Heating load | 0.996 | 0.630 | 0.469 |
| Random Forest | Cooling load | 0.972 | 1.618 | 1.002 |

For a slower Random Forest hyperparameter search:

```bash
uv run python train.py --tune-rf
```

Custom input/output locations:

```bash
uv run python train.py \
  --data ENB2012_data.xlsx \
  --artifacts-dir artifacts \
  --reports-dir reports
```

## Run the Metrics Exporter Locally

The exporter trains the models automatically if reports do not exist yet, then exposes metrics for Prometheus.

```bash
uv run python metrics_server.py --port 8000
```

Open:

```text
http://localhost:8000/metrics
```

Important exported metrics:

| Metric | Meaning |
| --- | --- |
| `energy_model_r2{model,target}` | R2 score by model and target |
| `energy_model_rmse{model,target}` | RMSE by model and target |
| `energy_model_mae{model,target}` | MAE by model and target |
| `energy_feature_importance{feature}` | ExtraTrees feature importance |
| `energy_dataset_rows` | Number of dataset rows |
| `energy_dataset_columns` | Number of dataset columns |
| `energy_target_mean{target}` | Mean target value |
| `energy_target_std{target}` | Target standard deviation |

## Run Grafana and Prometheus with Docker

Start the complete monitoring stack:

```bash
docker compose up --build
```

Or run it in the background:

```bash
docker compose up --build -d
```

Services:

| Service | URL |
| --- | --- |
| Grafana | `http://localhost:3000` |
| Prometheus | `http://localhost:9090` |
| Metrics exporter | `http://localhost:8000/metrics` |

Grafana login:

```text
username: admin
password: admin
```

Dashboard:

```text
http://localhost:3000/d/energy-efficiency/energy-efficiency-model-dashboard
```

The dashboard is provisioned automatically under:

```text
Energy Efficiency / Energy Efficiency Model Dashboard
```

## Grafana Dashboard

The dashboard includes:

- Best R2 KPI
- Lowest RMSE KPI
- Dataset row and feature counts
- Model R2 comparison
- Model RMSE comparison
- Model MAE table
- Feature importance ranking
- Target distribution summary

Grafana provisioning files live in:

```text
monitoring/grafana/provisioning/
monitoring/grafana/dashboards/energy-efficiency.json
```

Prometheus configuration lives in:

```text
monitoring/prometheus/prometheus.yml
```

## Makefile Commands

The repository includes shortcut commands:

```bash
make train     # Train models and generate reports
make metrics   # Run local Prometheus metrics exporter
make grafana   # Start Docker Compose monitoring stack
make stop      # Stop Docker Compose stack
make clean     # Remove generated artifacts and reports
```

## Docker Notes for Windows and WSL

If you use Docker Desktop with WSL:

1. Start Docker Desktop.
2. Open Docker Desktop settings.
3. Go to Resources > WSL Integration.
4. Enable integration for your Linux distribution.
5. Restart the WSL terminal.
6. Check Docker access:

```bash
docker version
```

If Docker is visible but permission is denied, run the command from a terminal that has Docker socket access or adjust your Docker/WSL user permissions.

## Troubleshooting

If Grafana asks for a login, use:

```text
admin / admin
```

If you changed the local Grafana password and forgot it:

```bash
docker compose exec grafana grafana cli admin reset-admin-password admin
```

If the dashboard still shows an old version after edits:

```bash
docker compose restart grafana
```

Then hard refresh the browser tab with `Ctrl+F5`.

If Prometheus does not show targets:

```bash
docker compose ps
docker compose logs prometheus
docker compose logs metrics-exporter
```

If ports are already in use, stop the conflicting process or change the host ports in `docker-compose.yml`.

## Clean Up

Stop the Docker stack:

```bash
docker compose down
```

Remove containers and volumes:

```bash
docker compose down -v
```

Remove generated local training output:

```bash
make clean
```

## Reproducibility

The project uses fixed random seeds in the training code and a locked dependency file (`uv.lock`) to keep runs consistent.

Generated model artifacts and reports are intentionally ignored by Git:

```text
artifacts/
reports/
```

This keeps the repository lightweight while allowing anyone to regenerate the outputs from the included dataset.
