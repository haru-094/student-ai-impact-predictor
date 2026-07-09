"""Train and persist ML models."""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from src.features.build_features import build_preprocessor
from src.utils.helpers import load_config, get_logger, ROOT

log = get_logger(__name__)

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting_classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_all(X_train, y_train, cfg: dict, task: str = "regression") -> dict:
    """Train all models for the given task. Returns dict of fitted pipelines."""
    preprocessor = build_preprocessor(cfg)
    model_registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    model_names = cfg["models"][task]

    fitted = {}
    for name in model_names:
        log.info(f"Training [{task}] → {name} ...")
        pipe = build_pipeline(preprocessor, model_registry[name])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        log.info(f"  ✓ {name} done.")
    return fitted


def save_models(fitted: dict, cfg: dict, task: str) -> None:
    """Persist fitted pipelines to disk."""
    out_dir = ROOT / cfg["paths"]["models"] / task
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pipe in fitted.items():
        path = out_dir / f"{name}.pkl"
        joblib.dump(pipe, path)
        log.info(f"Saved: {path}")
