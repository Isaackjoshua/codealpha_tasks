from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.config import ProjectPaths
from utils.metrics import compute_classification_metrics
from utils.plots import plot_confusion_matrix, plot_feature_importance, plot_roc_curves


def _get_feature_names(pipeline) -> list[str]:
    # pipeline: features -> preprocess -> variance -> model
    preprocessor = pipeline.named_steps["preprocess"]
    selector = pipeline.named_steps.get("variance")
    try:
        names = list(preprocessor.get_feature_names_out())
        if selector is not None and hasattr(selector, "get_support"):
            mask = selector.get_support()
            if len(mask) == len(names):
                names = [n for n, keep in zip(names, mask) if keep]
        return names
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    paths = ProjectPaths.from_cwd()

    parser = argparse.ArgumentParser(description="Evaluate trained credit scoring models on the test set.")
    parser.add_argument("--splits", type=str, default=str(paths.data_dir / "splits.joblib"), help="Path to saved splits.")
    args = parser.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits file: {splits_path}. Run `python train.py` first.")

    splits = joblib.load(splits_path)
    x_test = splits["x_test"]
    y_test = splits["y_test"]

    model_paths = {
        "logistic_regression": paths.models_dir / "logistic_regression.joblib",
        "decision_tree": paths.models_dir / "decision_tree.joblib",
        "random_forest": paths.models_dir / "random_forest.joblib",
    }

    metrics_rows = []
    probas = {}
    preds = {}

    for name, model_path in model_paths.items():
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probas[name] = proba
        preds[name] = pred

        m = compute_classification_metrics(y_test.to_numpy(), pred, proba)
        metrics_rows.append(
            {
                "model": name,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
            }
        )

    if not metrics_rows:
        raise FileNotFoundError("No trained models found. Run `python train.py` first.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="roc_auc", ascending=False)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.reports_dir / "metrics_test.json"
    metrics_path.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2))

    print("Test-set metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")

    # ROC curves comparison
    plot_roc_curves(
        y_true=y_test.to_numpy(),
        model_probas={k: v for k, v in probas.items()},
        out_path=paths.figures_dir / "roc_curves.png",
    )
    print(f"Saved: {paths.figures_dir / 'roc_curves.png'}")

    # Confusion matrices
    for name, pred in preds.items():
        plot_confusion_matrix(
            y_true=y_test.to_numpy(),
            y_pred=pred,
            out_path=paths.figures_dir / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix - {name}",
        )

    # Feature importance for tree-based models
    for name in ["decision_tree", "random_forest"]:
        model_path = model_paths.get(name)
        if not model_path or not model_path.exists():
            continue
        model = joblib.load(model_path)
        estimator = model.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            continue

        feature_names = _get_feature_names(model)
        importances = estimator.feature_importances_
        if feature_names and len(feature_names) == len(importances):
            plot_feature_importance(
                feature_names=feature_names,
                importances=importances,
                out_path=paths.figures_dir / f"feature_importance_{name}.png",
                title=f"Feature Importance - {name}",
                top_k=20,
            )
            print(f"Saved: {paths.figures_dir / f'feature_importance_{name}.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
