# Handwritten Character Recognition (MNIST + EMNIST)

Deep-learning project for recognizing handwritten digits and alphabets from images using a CNN in TensorFlow/Keras.

## Project structure

```
data/
models/
utils/
train.py
evaluate.py
inference.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you hit `ModuleNotFoundError: No module named 'importlib_resources'` on Python 3.12, reinstall requirements (this project includes `importlib-resources` to satisfy TFDS runtime imports).

## Datasets

- **MNIST**: handwritten digits (10 classes).
- **EMNIST**: handwritten characters. This project supports:
  - `emnist/byclass` (62 classes: digits + uppercase + lowercase)
  - `emnist/balanced` (47 classes)

Datasets are downloaded automatically on first run via `tensorflow_datasets`.

## Train

Train on MNIST:
```bash
python train.py --dataset mnist --run_dir models/mnist_cnn --epochs 20 --batch_size 128 --augment
```

Train on EMNIST (byclass):
```bash
python train.py --dataset emnist/byclass --run_dir models/emnist_byclass_cnn --epochs 30 --batch_size 256 --augment
```

Outputs (saved under `--run_dir`):
- `model.keras` (best model checkpoint)
- `meta.json` (class names + preprocessing metadata)
- `history.json` (training curves)
- `figures/` (loss/accuracy curves and sample predictions)

## Evaluate

```bash
python evaluate.py --run_dir models/emnist_byclass_cnn
```

Prints:
- test accuracy
- confusion matrix
- classification report

Also saves figures under `models/.../figures/`.

## Inference on an external image

```bash
python inference.py --run_dir models/mnist_cnn --image path/to/image.png
```

## Streamlit UI

Run a simple web UI for uploading/drawing digits and characters:

```bash
streamlit run app.py
```

In the sidebar, set **Run directory** to something like `models/mnist_cnn` (it must contain `model.keras` and `meta.json`).

Notes:
- The input image can be RGB or grayscale; it will be converted to grayscale.
- It is resized to 28×28 and normalized to `[0, 1]`.
- If the background appears white, the image is automatically inverted to match MNIST/EMNIST style.

## Extending to word/sentence recognition (CRNN + CTC)

Single-character classification assumes **one label per image**. For words/sentences, you typically need **sequence modeling**:

- **CRNN (Convolutional Recurrent Neural Network)**:
  - CNN backbone extracts a sequence of feature vectors across the image width.
  - An RNN (LSTM/GRU, often bidirectional) models dependencies across that sequence.
- **CTC loss (Connectionist Temporal Classification)**:
  - Trains without frame-level alignment between characters and image columns.
  - Decodes the most likely character sequence using greedy decoding or beam search.

High-level approach:
1. Use a CNN to transform the image into a feature map.
2. Collapse height (e.g., via pooling) so width becomes the time dimension.
3. Feed the width-wise sequence to BiLSTM/GRU layers.
4. Predict per-timestep character probabilities.
5. Train with **CTC loss** and decode sequences at inference time.

This is the standard recipe used in many OCR systems for word-level recognition.
