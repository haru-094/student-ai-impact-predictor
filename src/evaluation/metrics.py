"""Compute and display model evaluation metrics."""

from typing import List, Dict
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from src.utils.helpers import get_logger
import numpy as np

log = get_logger(__name__)


def evaluate_regression(y_true, y_pred, model_name: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    result = {"model": model_name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
    log.info(f"[Regression] {model_name} → MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    return result


def evaluate_classification(y_true, y_pred, model_name: str = "") -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    result = {"model": model_name, "Accuracy": round(acc, 4), "F1_weighted": round(f1, 4)}
    log.info(f"[Classification] {model_name} → Accuracy={acc:.4f} | F1={f1:.4f}")
    log.info(f"\n{classification_report(y_true, y_pred)}")
    return result


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """Return a sorted DataFrame from a list of metric dicts."""
    return pd.DataFrame(results)
