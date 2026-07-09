"""
Entry point — runs the full ML pipeline end-to-end.

Usage:
    uv run python main.py
    uv run python main.py --task regression
    uv run python main.py --task classification
"""

import argparse
from src.utils.helpers import load_config, get_logger, ensure_dirs
from src.data.ingest import load_raw
from src.data.preprocess import clean, split
from src.models.train import train_all, save_models
from src.evaluation.metrics import (
    evaluate_regression, evaluate_classification, compare_models
)
from src.visualization.visualize import plot_target_distribution, plot_correlation_heatmap, plot_model_comparison

log = get_logger("main")


def run(task: str = "regression") -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    # 1. Load & clean
    df = load_raw()
    df = clean(df, cfg)

    # 2. Select target
    if task == "regression":
        target = cfg["data"]["target_regression"]
    else:
        target = cfg["data"]["target_classification"]

    # 3. Visualise
    plot_target_distribution(df[target], cfg)
    plot_correlation_heatmap(df, cfg)

    # 4. Split
    X_train, X_test, y_train, y_test = split(df, target, cfg)

    # 5. Train
    fitted = train_all(X_train, y_train, cfg, task=task)

    # 6. Evaluate
    results = []
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        if task == "regression":
            metrics = evaluate_regression(y_test, y_pred, model_name=name)
        else:
            metrics = evaluate_classification(y_test, y_pred, model_name=name)
        results.append(metrics)

    # 7. Compare & visualise
    results_df = compare_models(results)
    print("\n── Model Comparison ──")
    print(results_df.to_string(index=False))

    metric = "R2" if task == "regression" else "F1_weighted"
    plot_model_comparison(results_df, metric, cfg)

    # 8. Persist best model
    save_models(fitted, cfg, task)
    log.info("Pipeline complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    args = parser.parse_args()
    run(task=args.task)
