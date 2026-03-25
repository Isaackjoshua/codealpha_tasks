from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


def preprocess_pil_image(
    img: Image.Image,
    *,
    target_size: Tuple[int, int] = (28, 28),
    invert_if_needed: bool = True,
) -> np.ndarray:
    """
    Preprocess a PIL image to match MNIST/EMNIST model input.

    Returns: array of shape (1, 28, 28, 1)
    """
    img = img.convert("L")
    img = ImageOps.fit(img, target_size, method=Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32) / 255.0
    if invert_if_needed and float(x.mean()) > 0.5:
        x = 1.0 - x

    x = np.clip(x, 0.0, 1.0)
    x = np.expand_dims(x, axis=-1)  # (28, 28, 1)
    x = np.expand_dims(x, axis=0)  # (1, 28, 28, 1)
    return x


def load_and_preprocess_image(
    image_path: str | Path,
    *,
    target_size: Tuple[int, int] = (28, 28),
    invert_if_needed: bool = True,
) -> np.ndarray:
    """
    Loads an external image and preprocesses it to match MNIST/EMNIST model input:
    - convert to grayscale
    - resize to 28x28
    - normalize to [0, 1]
    - add channel dimension

    Returns: array of shape (1, 28, 28, 1)
    """
    img = Image.open(image_path)
    return preprocess_pil_image(img, target_size=target_size, invert_if_needed=invert_if_needed)

