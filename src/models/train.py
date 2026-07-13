"""Train and persist ML models."""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

REGRESSION_GRIDS = {
    "linear_regression": {},
    "random_forest_regressor": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 15, None],
    },
    "gradient_boosting_regressor": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
    },
}

CLASSIFICATION_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest_classifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
    },
    "gradient_boosting_classifier": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
    },
}


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_all(X_train, y_train, cfg: dict, task: str = "regression", tune: bool = False) -> dict:
    """Train all models for the given task. Returns dict of fitted pipelines."""
    preprocessor = build_preprocessor(cfg)
    model_registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    model_names = cfg["models"][task]

    tuning_method = cfg.get("tuning", {}).get("method", "grid_search")
    cv_folds = cfg.get("tuning", {}).get("cv_folds", 5)

    if task == "regression":
        scoring = cfg.get("tuning", {}).get("scoring_regression", "neg_root_mean_squared_error")
        grids = REGRESSION_GRIDS
    else:
        scoring = cfg.get("tuning", {}).get("scoring_classification", "f1_weighted")
        grids = CLASSIFICATION_GRIDS

    fitted = {}
    for name in model_names:
        pipe = build_pipeline(preprocessor, model_registry[name])
        param_grid = grids.get(name, {})

        if tune and param_grid:
            log.info(f"Tuning [{task}] → {name} using {tuning_method} with {cv_folds}-fold CV...")
            if tuning_method == "grid_search":
                search = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1)
            else:
                search = RandomizedSearchCV(pipe, param_grid, n_iter=8, cv=cv_folds, scoring=scoring, n_jobs=-1, random_state=42, verbose=1)
            
            search.fit(X_train, y_train)
            log.info(f"  ✓ {name} best parameters: {search.best_params_}")
            log.info(f"  ✓ {name} best CV score ({scoring}): {search.best_score_:.4f}")
            fitted[name] = search.best_estimator_
        else:
            log.info(f"Training [{task}] → {name} ...")
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
