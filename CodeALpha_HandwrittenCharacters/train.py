from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from data.datasets import load_dataset
from models.cnn import build_cnn
from utils.io import ensure_dir, write_json
from utils.plotting import plot_history
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on MNIST or EMNIST.")
    p.add_argument("--dataset", type=str, default="mnist", help="mnist | emnist/byclass | emnist/balanced")
    p.add_argument("--run_dir", type=str, default="models/run", help="Output directory for model + artifacts")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--augment", action="store_true", help="Enable rotation/zoom augmentation")
    p.add_argument("--noise", action="store_true", help="Add Gaussian noise augmentation during training")
    p.add_argument(
        "--fix_emnist_orientation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply transpose+flip fix for EMNIST images (default: on for EMNIST, off for MNIST)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_dir = ensure_dir(args.run_dir)
    figs_dir = ensure_dir(run_dir / "figures")

    bundle = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        augment=args.augment,
        add_noise=args.noise,
        fix_emnist_orientation=args.fix_emnist_orientation,
        data_dir="data",
    )

    model = build_cnn(bundle.input_shape, bundle.num_classes, dropout_rate=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    ckpt_path = str(run_dir / "model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save metadata needed for evaluation/inference.
    write_json(
        run_dir / "meta.json",
        {
            "dataset_name": bundle.dataset_name,
            "input_shape": list(bundle.input_shape),
            "num_classes": bundle.num_classes,
            "class_names": bundle.class_names,
            "preprocessing": {
                "normalize_0_1": True,
                "resize": [28, 28],
                "channels": 1,
                "fix_emnist_orientation": args.fix_emnist_orientation,
            },
        },
    )

    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    write_json(run_dir / "history.json", hist_dict)
    plot_history(hist_dict, figs_dir / "training_curves.png")

    # Ensure the final model is saved even if ModelCheckpoint didn't trigger for some reason.
    model.save(ckpt_path)

    print(f"Saved model to: {ckpt_path}")
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
