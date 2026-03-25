from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.image import load_and_preprocess_image
from utils.io import read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on an external image.")
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing model.keras and meta.json")
    p.add_argument("--image", type=str, required=True, help="Path to an input image file")
    p.add_argument("--topk", type=int, default=5, help="Show top-K predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    meta = read_json(run_dir / "meta.json")
    class_names = meta["class_names"]

    model = tf.keras.models.load_model(run_dir / "model.keras")
    x = load_and_preprocess_image(args.image, target_size=(28, 28), invert_if_needed=True)

    probs = model.predict(x, verbose=0)[0]
    topk = int(max(1, min(args.topk, len(probs))))
    idx = np.argsort(-probs)[:topk]

    best = int(idx[0])
    print(f"Prediction: {class_names[best]} (p={float(probs[best]):.4f})")
    if topk > 1:
        print("\nTop predictions:")
        for i in idx:
            print(f"- {class_names[int(i)]}: {float(probs[int(i)]):.4f}")


if __name__ == "__main__":
    main()

