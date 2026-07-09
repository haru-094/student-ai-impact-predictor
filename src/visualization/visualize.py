"""Common plotting helpers."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.utils.helpers import get_logger, ROOT

log = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def save_fig(fig, name: str, cfg: dict) -> None:
    out = ROOT / cfg["paths"]["figures"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Saved figure: {out}")
    plt.close(fig)


def plot_target_distribution(series: pd.Series, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Distribution of {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    save_fig(fig, f"distribution_{series.name}.png", cfg)


def plot_correlation_heatmap(df: pd.DataFrame, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_fig(fig, "correlation_heatmap.png", cfg)


def plot_model_comparison(results_df: pd.DataFrame, metric: str, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.sort_values(metric, ascending=False).plot(
        x="model", y=metric, kind="bar", ax=ax, color="teal", edgecolor="white"
    )
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    save_fig(fig, f"model_comparison_{metric}.png", cfg)
