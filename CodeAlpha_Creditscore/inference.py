from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from utils.config import ProjectPaths, RAW_FEATURES


@dataclass(frozen=True)
class InferenceResult:
    label: str
    probability_creditworthy: float


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with `python train.py`.")
    return joblib.load(model_path)


def predict_creditworthiness(model, user_features: Dict[str, Any]) -> InferenceResult:
    missing = [c for c in RAW_FEATURES if c not in user_features]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    x = pd.DataFrame([user_features], columns=RAW_FEATURES)
    proba = float(model.predict_proba(x)[:, 1][0])
    label = "good" if proba >= 0.5 else "bad"
    return InferenceResult(label=label, probability_creditworthy=proba)


def _prompt_float(name: str) -> float:
    while True:
        raw = input(f"{name}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(name: str) -> int:
    return int(_prompt_float(name))


def prompt_user_input() -> Dict[str, Any]:
    print("Enter applicant details:")
    income = _prompt_float("income (annual, e.g., 55000)")
    age = _prompt_int("age")
    employment_status = input("employment_status (employed/self_employed/unemployed/student/retired): ").strip()
    debt_amount = _prompt_float("debt_amount")
    credit_history_length_years = _prompt_float("credit_history_length_years")
    on_time_payment_ratio = _prompt_float("on_time_payment_ratio (0-1)")
    late_payments_count = _prompt_int("late_payments_count")
    open_accounts = _prompt_int("open_accounts")
    credit_utilization_ratio = _prompt_float("credit_utilization_ratio (0-1)")

    return {
        "income": income,
        "age": age,
        "employment_status": employment_status,
        "debt_amount": debt_amount,
        "credit_history_length_years": credit_history_length_years,
        "on_time_payment_ratio": on_time_payment_ratio,
        "late_payments_count": late_payments_count,
        "open_accounts": open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
    }


def main() -> int:
    paths = ProjectPaths.from_cwd()

    parser = argparse.ArgumentParser(description="Run inference using the saved best credit scoring model.")
    parser.add_argument("--model", type=str, default=str(paths.models_dir / "best_model.joblib"), help="Path to model.")
    parser.add_argument("--json", type=str, default="", help="JSON string with applicant features.")
    args = parser.parse_args()

    model = load_model(Path(args.model))

    if args.json:
        payload = json.loads(args.json)
        if not isinstance(payload, dict):
            raise ValueError("--json must be a JSON object.")
        user_features = payload
    else:
        user_features = prompt_user_input()

    result = predict_creditworthiness(model, user_features)
    print(json.dumps({"prediction": result.label, "probability_creditworthy": result.probability_creditworthy}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

