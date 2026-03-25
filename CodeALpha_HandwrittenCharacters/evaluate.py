from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data.datasets import load_dataset
from utils.io import ensure_dir, read_json, write_json
from utils.metrics import compute_metrics
from utils.plotting import plot_confusion_matrix, plot_sample_predictions
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained handwritten CNN.")
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing model.keras and meta.json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_plot_classes", type=int, default=40, help="Max classes to plot for confusion matrix")
    p.add_argument(
        "--fix_emnist_orientation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override EMNIST orientation fix (default: use value from meta.json if present)",
    )
    return p.parse_args()


def _collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    images = []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        pred = np.argmax(probs, axis=1)
        true = np.argmax(batch_y.numpy(), axis=1)

        y_true.append(true)
        y_pred.append(pred)
        images.append(batch_x.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    images = np.concatenate(images, axis=0)
    return images, y_true, y_pred


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_dir = Path(args.run_dir)
    meta = read_json(run_dir / "meta.json")
    class_names = meta["class_names"]
    meta_fix = meta.get("preprocessing", {}).get("fix_emnist_orientation", None)
    fix_emnist_orientation = args.fix_emnist_orientation if args.fix_emnist_orientation is not None else meta_fix

    model = tf.keras.models.load_model(run_dir / "model.keras")

    bundle = load_dataset(
        meta["dataset_name"],
        batch_size=args.batch_size,
        val_split=0.1,  # not used for test; included for API compatibility
        seed=args.seed,
        augment=False,
        add_noise=False,
        fix_emnist_orientation=fix_emnist_orientation,
        data_dir="data",
    )

    figs_dir = ensure_dir(run_dir / "figures")

    images, y_true, y_pred = _collect_predictions(model, bundle.test)
    metrics = compute_metrics(y_true, y_pred, class_names)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    cm = metrics["confusion_matrix"]
    plot_confusion_matrix(cm, class_names, figs_dir / "confusion_matrix.png", max_classes=args.max_plot_classes)

    # Save sample predictions figure
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(images), size=min(25, len(images)), replace=False)
    plot_sample_predictions(
        images[idx],
        y_true[idx],
        y_pred[idx],
        class_names,
        figs_dir / "sample_predictions.png",
        n=min(25, len(idx)),
    )

    # Persist evaluation metrics for reproducibility.
    write_json(
        run_dir / "eval.json",
        {
            "accuracy": metrics["accuracy"],
        },
    )

    print(f"\nSaved figures to: {figs_dir}")


if __name__ == "__main__":
    main()
