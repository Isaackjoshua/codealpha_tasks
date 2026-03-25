from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils.image import preprocess_pil_image
from utils.io import read_json


def load_run(run_dir: Path) -> tuple[tf.keras.Model, dict]:
    meta_path = run_dir / "meta.json"
    model_path = run_dir / "model.keras"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")

    meta = read_json(meta_path)
    model = tf.keras.models.load_model(model_path)
    return model, meta


def predict_topk(model: tf.keras.Model, x: np.ndarray, class_names: list[str], topk: int) -> list[tuple[str, float]]:
    probs = model.predict(x, verbose=0)[0]
    topk = int(max(1, min(topk, len(probs))))
    idx = np.argsort(-probs)[:topk]
    return [(class_names[int(i)], float(probs[int(i)])) for i in idx]


def main() -> None:
    st.set_page_config(page_title="Handwritten Character Recognition", layout="wide")
    st.title("Handwritten Character Recognition (MNIST / EMNIST)")

    with st.sidebar:
        st.header("Model")
        default_run = "models/mnist_cnn"
        run_dir_str = st.text_input("Run directory", value=st.session_state.get("run_dir", default_run))
        topk = st.slider("Top-K", min_value=1, max_value=10, value=5)
        load_clicked = st.button("Load model")

        if load_clicked or ("model" not in st.session_state):
            try:
                run_dir = Path(run_dir_str)
                model, meta = load_run(run_dir)
                st.session_state["model"] = model
                st.session_state["meta"] = meta
                st.session_state["run_dir"] = str(run_dir)
                st.success("Loaded.")
            except Exception as e:
                st.session_state.pop("model", None)
                st.session_state.pop("meta", None)
                st.error(str(e))

        if "meta" in st.session_state:
            meta = st.session_state["meta"]
            st.caption(f"Dataset: `{meta.get('dataset_name', 'unknown')}`")
            st.caption(f"Classes: `{meta.get('num_classes', 'unknown')}`")

    if "model" not in st.session_state:
        st.info("Load a trained run directory that contains `model.keras` and `meta.json`.")
        st.stop()

    model: tf.keras.Model = st.session_state["model"]
    meta: dict = st.session_state["meta"]
    class_names: list[str] = meta["class_names"]

    tab_infer, tab_about = st.tabs(["Inference", "About"])

    with tab_infer:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Input")
            mode = st.radio("Choose input type", options=["Upload image", "Draw"], horizontal=True)

            pil_img: Image.Image | None = None
            invert_if_needed = True

            if mode == "Upload image":
                uploaded = st.file_uploader("Upload a PNG/JPG image", type=["png", "jpg", "jpeg"])
                invert_if_needed = st.checkbox("Auto-invert if background looks white", value=True)
                if uploaded is not None:
                    pil_img = Image.open(uploaded)
                    st.image(pil_img, caption="Uploaded image", use_container_width=True)

            else:
                st.caption("Draw with white strokes on a black background.")
                canvas = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                if canvas.image_data is not None:
                    # RGBA -> grayscale
                    arr = (canvas.image_data[:, :, :3]).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                    invert_if_needed = False  # already matches MNIST style

        with col_right:
            st.subheader("Prediction")
            if pil_img is None:
                st.info("Provide an image (upload or draw) to see predictions.")
            else:
                x = preprocess_pil_image(pil_img, target_size=(28, 28), invert_if_needed=invert_if_needed)
                preds = predict_topk(model, x, class_names, topk=topk)

                st.metric("Predicted", preds[0][0], delta=f"p={preds[0][1]:.4f}")

                st.caption("Top-K probabilities")
                st.bar_chart({label: prob for label, prob in preds}, horizontal=True)

                st.caption("Preprocessed (28×28)")
                st.image(x[0].squeeze(), clamp=True, channels="GRAY", use_container_width=False)

    with tab_about:
        st.markdown(
            """
This UI loads a trained run directory (containing `model.keras` and `meta.json`) and runs inference on:

- uploaded images (PNG/JPG)
- drawings from a canvas

Training and evaluation are done via:
- `train.py`
- `evaluate.py`
- `inference.py`
"""
        )


if __name__ == "__main__":
    main()

