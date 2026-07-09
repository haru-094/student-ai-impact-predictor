"""Load raw data from disk."""

import pandas as pd
from pathlib import Path
from src.utils.helpers import load_config, get_logger, ROOT

log = get_logger(__name__)


def load_raw(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    cfg = load_config(config_path)
    raw_path = ROOT / cfg["paths"]["raw_data"]
    log.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    log.info(f"Shape: {df.shape} | Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
