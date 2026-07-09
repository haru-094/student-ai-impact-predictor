"""Load a saved model and generate predictions."""

import joblib
import pandas as pd
from pathlib import Path
from src.utils.helpers import load_config, get_logger, ROOT

log = get_logger(__name__)


def load_model(model_name: str, task: str, cfg: dict):
    path = ROOT / cfg["paths"]["models"] / task / f"{model_name}.pkl"
    log.info(f"Loading model from: {path}")
    return joblib.load(path)


def predict(model, X: pd.DataFrame):
    return model.predict(X)
