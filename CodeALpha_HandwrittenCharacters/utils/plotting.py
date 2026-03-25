from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: Dict[str, List[float]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    *,
    max_classes: int = 40,
) -> None:
    """
    For very large label sets (e.g., EMNIST/byclass=62), the plot can become unreadable.
    This function optionally truncates to the first max_classes classes for visualization.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_plot = cm
    names_plot = list(class_names)
    if len(names_plot) > max_classes:
        cm_plot = cm[:max_classes, :max_classes]
        names_plot = names_plot[:max_classes]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + ("" if len(class_names) <= max_classes else f" (first {max_classes})"))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(names_plot)))
    ax.set_yticks(np.arange(len(names_plot)))
    ax.set_xticklabels(names_plot, rotation=90, fontsize=7)
    ax.set_yticklabels(names_plot, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    *,
    n: int = 25,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(n, len(images))
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        img = images[i].squeeze()
        ax.imshow(img, cmap="gray")
        t = class_names[int(y_true[i])]
        p = class_names[int(y_pred[i])]
        ax.set_title(f"T:{t} P:{p}", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


