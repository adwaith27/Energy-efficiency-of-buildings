from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DATASET_COLUMNS = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

FEATURE_COLUMNS = list(DATASET_COLUMNS.values())[:8]
TARGET_COLUMNS = list(DATASET_COLUMNS.values())[8:]
CORRELATION_FEATURES = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "glazing_area",
]


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    target: str
    r2: float
    rmse: float
    mae: float


def load_dataset(path: str | Path = "ENB2012_data.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = set(DATASET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df.rename(columns=DATASET_COLUMNS)


def split_features_targets(
    df: pd.DataFrame,
    features: Iterable[str] = FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
):
    x = df[list(features)]
    y = df[TARGET_COLUMNS]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def evaluate_predictions(model_name: str, y_true: pd.DataFrame, y_pred: np.ndarray) -> list[EvaluationResult]:
    r2_values = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse_values = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mae_values = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    return [
        EvaluationResult(model_name, target, float(r2), float(rmse), float(mae))
        for target, r2, rmse, mae in zip(TARGET_COLUMNS, r2_values, rmse_values, mae_values)
    ]


def build_models(random_state: int = 42) -> dict[str, object]:
    polynomial = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=10.0))]),
        "lasso": Pipeline([("scaler", StandardScaler()), ("regressor", Lasso(alpha=0.01, max_iter=10000))]),
        "polynomial_regression": polynomial,
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def tune_random_forest(x_train: pd.DataFrame, y_train: pd.DataFrame, random_state: int = 42) -> RandomForestRegressor:
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.75, 1.0],
        "bootstrap": [True, False],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=24,
        cv=5,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def tune_linear_models(df: pd.DataFrame) -> dict[str, float]:
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    alphas = np.logspace(-4, 4, 9)
    ridge_scores = [cross_val_score(Ridge(alpha=alpha), x, y, cv=10, scoring="r2").mean() for alpha in alphas]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    lasso_search = GridSearchCV(
        estimator=Lasso(max_iter=10000),
        param_grid={"alpha": alphas},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
    )
    lasso_search.fit(x_scaled, y)
    return {
        "ridge_alpha": float(alphas[int(np.argmax(ridge_scores))]),
        "lasso_alpha": float(lasso_search.best_params_["alpha"]),
    }


def feature_importance(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    model = ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMNS])
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True).round(4)


def train_and_evaluate(
    data_path: str | Path = "ENB2012_data.xlsx",
    artifacts_dir: str | Path = "artifacts",
    reports_dir: str | Path = "reports",
    tune_rf: bool = False,
) -> pd.DataFrame:
    artifacts_path = Path(artifacts_dir)
    reports_path = Path(reports_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_features_targets(df)

    tuned = tune_linear_models(df)
    models = build_models()
    models["ridge"].set_params(regressor__alpha=tuned["ridge_alpha"])
    models["lasso"].set_params(regressor__alpha=tuned["lasso_alpha"])
    if tune_rf:
        models["random_forest"] = tune_random_forest(x_train, y_train)

    rows: list[EvaluationResult] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.extend(evaluate_predictions(name, y_test, predictions))
        joblib.dump(model, artifacts_path / f"{name}.joblib")

    metrics = pd.DataFrame([result.__dict__ for result in rows])
    metrics.to_csv(reports_path / "model_metrics.csv", index=False)
    feature_importance(df).to_csv(reports_path / "feature_importance.csv", index=False)
    correlation_table(df).to_csv(reports_path / "correlations.csv")
    pd.Series(tuned).to_json(reports_path / "tuned_parameters.json", indent=2)
    return metrics
