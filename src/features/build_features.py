"""Feature engineering + sklearn preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from src.utils.helpers import load_config, get_logger

log = get_logger(__name__)

# ── Ordinal mappings (derived from EDA) ──────────────────────────────────────
YEAR_ORDER = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4, "Graduate": 5}
SKILL_ORDER = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
USE_CASE_GPA_RANK = {
    "Direct_Answer_Generation": 1,
    "Copywriting/Drafting": 2,
    "Ideation": 3,
    "Summarizing_Reading": 4,
    "Debugging/Troubleshooting": 5,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features. Returns a copy with new columns."""
    df = df.copy()

    # ── Interaction / ratio features ─────────────────────────────────────────
    df["study_ratio"] = df["Traditional_Study_Hours"] / (df["Weekly_GenAI_Hours"] + 1)
    df["total_study_hours"] = df["Traditional_Study_Hours"] + df["Weekly_GenAI_Hours"]
    df["genai_dependency_score"] = df["Weekly_GenAI_Hours"] * df["Perceived_AI_Dependency"]
    df["ai_efficiency"] = df["Skill_Retention_Score"] / (df["Weekly_GenAI_Hours"] + 1)
    df["anxiety_gpa_pressure"] = df["Anxiety_Level_During_Exams"] * (4 - df["Pre_Semester_GPA"])

    # Target variable for GPA impact (re-framed to avoid target leakage)
    if "Post_Semester_GPA" in df.columns:
        df["gpa_change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]

    # ── Ordinal encodings ────────────────────────────────────────────────────
    df["Year_of_Study_enc"] = df["Year_of_Study"].map(YEAR_ORDER).astype(int)
    df["Prompt_Engineering_Skill_enc"] = df["Prompt_Engineering_Skill"].map(SKILL_ORDER).astype(int)
    df["Primary_Use_Case_enc"] = df["Primary_Use_Case"].map(USE_CASE_GPA_RANK).astype(int)

    # ── Binary: bool → int ───────────────────────────────────────────────────
    df["Paid_Subscription"] = df["Paid_Subscription"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).astype(int)

    log.info(f"Feature engineering done. Shape: {df.shape}")
    return df


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Return a ColumnTransformer that scales/encodes all features."""
    num_features = cfg["features"]["numerical"]
    ord_features = cfg["features"]["ordinal"]
    cat_features = cfg["features"]["categorical"]
    bin_features = cfg["features"]["binary"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Ordinals are already integers — just scale them
    ord_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("ord", ord_pipeline, ord_features),
        ("cat", cat_pipeline, cat_features),
        ("bin", "passthrough", bin_features),
    ])

    log.info("Preprocessor built.")
    return preprocessor
