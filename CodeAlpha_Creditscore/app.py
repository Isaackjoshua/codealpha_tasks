from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from inference import predict_creditworthiness
from utils.config import ProjectPaths, RAW_FEATURES


def _run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return proc.returncode, output


def _load_model(model_path: Path):
    import joblib

    return joblib.load(model_path)


def _predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    x = df[RAW_FEATURES].copy()
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["probability_creditworthy"] = proba
    out["prediction"] = pd.Series(pred).map({1: "good", 0: "bad"})
    return out


def main() -> None:
    paths = ProjectPaths.from_cwd()
    default_data = paths.data_dir / "credit_data.csv"
    default_model = paths.models_dir / "best_model.joblib"

    st.set_page_config(page_title="Credit Scoring System", layout="wide")
    st.title("Credit Scoring System")
    st.caption("Predict creditworthiness (good/bad) from financial + behavioral features.")

    with st.sidebar:
        st.header("Artifacts")
        data_path = Path(st.text_input("Dataset CSV path", value=str(default_data)))
        model_path = Path(st.text_input("Model path", value=str(default_model)))

        st.divider()
        st.subheader("Training")
        st.write("If the model does not exist yet, train it from the CLI or use the button below.")
        if st.button("Generate data + Train models", type="primary"):
            cmd = [sys.executable, "train.py", "--data", str(data_path)]
            code, out = _run_command(cmd, cwd=paths.root)
            st.session_state["train_log"] = out
            if code == 0:
                st.success("Training completed. Reload the app if needed.")
            else:
                st.error("Training failed. See logs below.")
        if "train_log" in st.session_state:
            st.text_area("Training logs", value=st.session_state["train_log"], height=260)

    tab_predict, tab_batch, tab_data = st.tabs(["Single Prediction", "Batch Prediction", "Data Preview"])

    with tab_predict:
        st.subheader("Single applicant")
        col1, col2, col3 = st.columns(3)

        with col1:
            income = st.number_input("Income (annual)", min_value=0.0, value=55000.0, step=1000.0)
            age = st.number_input("Age", min_value=18, max_value=100, value=34, step=1)
            employment_status = st.selectbox(
                "Employment status",
                options=["employed", "self_employed", "unemployed", "student", "retired"],
                index=0,
            )

        with col2:
            debt_amount = st.number_input("Debt amount", min_value=0.0, value=12000.0, step=500.0)
            credit_history_length_years = st.number_input(
                "Credit history length (years)", min_value=0.0, value=8.0, step=0.5
            )
            open_accounts = st.number_input("Open accounts", min_value=0, value=6, step=1)

        with col3:
            on_time_payment_ratio = st.slider("On-time payment ratio", min_value=0.0, max_value=1.0, value=0.92, step=0.01)
            late_payments_count = st.number_input("Late payments count", min_value=0, value=1, step=1)
            credit_utilization_ratio = st.slider(
                "Credit utilization ratio", min_value=0.0, max_value=1.0, value=0.28, step=0.01
            )

        payload: Dict[str, Any] = {
            "income": float(income),
            "age": int(age),
            "employment_status": str(employment_status),
            "debt_amount": float(debt_amount),
            "credit_history_length_years": float(credit_history_length_years),
            "on_time_payment_ratio": float(on_time_payment_ratio),
            "late_payments_count": int(late_payments_count),
            "open_accounts": int(open_accounts),
            "credit_utilization_ratio": float(credit_utilization_ratio),
        }

        st.code(json.dumps(payload, indent=2), language="json")

        if st.button("Predict", type="primary"):
            if not model_path.exists():
                st.error(f"Model not found at `{model_path}`. Train first using the sidebar button or `python train.py`.")
            else:
                model = _load_model(model_path)
                result = predict_creditworthiness(model, payload)
                st.metric("Prediction", result.label)
                st.metric("P(creditworthy)", f"{result.probability_creditworthy:.3f}")

    with tab_batch:
        st.subheader("Batch prediction from CSV")
        st.write("Upload a CSV containing the required feature columns.")
        uploaded = st.file_uploader("CSV file", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("Run batch prediction", type="primary"):
                if not model_path.exists():
                    st.error(f"Model not found at `{model_path}`. Train first.")
                else:
                    model = _load_model(model_path)
                    try:
                        scored = _predict_batch(model, df)
                        st.success("Scoring completed.")
                        st.dataframe(scored.head(50), use_container_width=True)
                        st.download_button(
                            "Download scored CSV",
                            data=scored.to_csv(index=False).encode("utf-8"),
                            file_name="scored_credit_data.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

        st.info(f"Required columns: {', '.join(RAW_FEATURES)}")

    with tab_data:
        st.subheader("Dataset preview")
        if data_path.exists():
            df = pd.read_csv(data_path)
            st.write(f"Loaded `{data_path}` — shape: {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(50), use_container_width=True)
            st.write("Missing values per column:")
            st.dataframe(df.isna().sum().sort_values(ascending=False), use_container_width=True)
        else:
            st.warning(f"No dataset found at `{data_path}`. Train to generate one, or upload via the Batch tab.")


if __name__ == "__main__":
    main()

