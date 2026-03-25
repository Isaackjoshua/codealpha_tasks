from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_roc_curves(
    y_true: np.ndarray,
    model_probas: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "ROC Curves (Test Set)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for name, proba in model_probas.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    title: str = "Feature Importance",
    top_k: int = 20,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    plt.figure(figsize=(8, max(4.5, 0.28 * len(top_features))))
    sns.barplot(x=top_importances, y=top_features, orient="h")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

