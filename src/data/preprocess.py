"""Clean and split the dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.helpers import load_config, get_logger, ROOT

log = get_logger(__name__)


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop irrelevant columns and handle missing values."""
    drop_cols = cfg["data"].get("drop_columns", [])
    df = df.drop(columns=drop_cols, errors="ignore")
    log.info(f"Dropped columns: {drop_cols}")

    # Convert boolean-like strings
    if "Paid_Subscription" in df.columns:
        df["Paid_Subscription"] = df["Paid_Subscription"].map(
            {True: 1, False: 0, "True": 1, "False": 0}
        )

    before = len(df)
    df = df.dropna()
    log.info(f"Dropped {before - len(df)} rows with NaN. Remaining: {len(df)}")
    return df


def split(
    df: pd.DataFrame,
    target: str,
    cfg: dict,
    random_state: int = 42,
):
    """Return (X_train, X_test, y_train, y_test)."""
    X = df.drop(columns=[target])
    y = df[target]
    test_size = cfg["data"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log.info(
        f"Split → train: {len(X_train)} | test: {len(X_test)} | target: '{target}'"
    )
    return X_train, X_test, y_train, y_test
