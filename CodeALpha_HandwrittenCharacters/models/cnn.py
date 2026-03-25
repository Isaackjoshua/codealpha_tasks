from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    *,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Simple CNN for 28x28 grayscale handwritten character classification.
    Architecture matches the suggested template in the prompt.
    """
    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="handwritten_cnn")


