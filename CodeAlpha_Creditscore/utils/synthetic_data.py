from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import RANDOM_STATE, RAW_FEATURES, TARGET_COL


@dataclass(frozen=True)
class SyntheticDataConfig:
    n_rows: int = 5000
    missing_rate: float = 0.03
    outlier_rate: float = 0.01
    random_state: int = RANDOM_STATE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_credit_data(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)

    employment_statuses = np.array(
        ["employed", "self_employed", "unemployed", "student", "retired"], dtype=object
    )
    employment = rng.choice(employment_statuses, size=config.n_rows, p=[0.6, 0.15, 0.1, 0.1, 0.05])

    age = rng.integers(18, 75, size=config.n_rows)
    income = rng.lognormal(mean=10.5, sigma=0.5, size=config.n_rows)  # ~36k median-ish

    employment_income_multiplier = np.select(
        [
            employment == "employed",
            employment == "self_employed",
            employment == "unemployed",
            employment == "student",
            employment == "retired",
        ],
        [1.00, 1.10, 0.35, 0.50, 0.70],
        default=1.0,
    )
    income = income * employment_income_multiplier

    credit_history_length_years = np.clip((age - 18) * rng.uniform(0.2, 0.8, size=config.n_rows), 0, 40)
    open_accounts = np.clip(rng.poisson(lam=4, size=config.n_rows) + (age > 30).astype(int), 0, 25)
    credit_utilization_ratio = np.clip(rng.beta(a=2.5, b=5.0, size=config.n_rows), 0, 1)

    debt_amount = (
        income
        * (0.05 + 0.8 * credit_utilization_ratio)
        * rng.lognormal(mean=0.0, sigma=0.35, size=config.n_rows)
    )

    on_time_payment_ratio = np.clip(
        rng.beta(a=8.0, b=2.0, size=config.n_rows) - 0.5 * credit_utilization_ratio, 0, 1
    )
    late_payments_count = np.clip(
        rng.poisson(lam=1.5 + 6.0 * (1 - on_time_payment_ratio), size=config.n_rows), 0, 50
    )

    # Latent creditworthiness score (log-odds)
    debt_to_income = debt_amount / np.maximum(income, 1.0)
    log_income = np.log(np.maximum(income, 1.0))
    util = credit_utilization_ratio

    employment_effect = np.select(
        [
            employment == "employed",
            employment == "self_employed",
            employment == "unemployed",
            employment == "student",
            employment == "retired",
        ],
        [0.35, 0.20, -0.75, -0.35, -0.10],
        default=0.0,
    )

    z = (
        0.85 * log_income
        - 3.0 * debt_to_income
        - 1.75 * util
        + 2.5 * on_time_payment_ratio
        - 0.12 * late_payments_count
        + 0.03 * credit_history_length_years
        + 0.02 * open_accounts
        + employment_effect
        - 7.5  # shift to make realistic base rate
    )
    p_good = _sigmoid(z)
    y = rng.binomial(n=1, p=p_good, size=config.n_rows)

    df = pd.DataFrame(
        {
            "income": income,
            "age": age,
            "employment_status": employment,
            "debt_amount": debt_amount,
            "credit_history_length_years": credit_history_length_years,
            "on_time_payment_ratio": on_time_payment_ratio,
            "late_payments_count": late_payments_count,
            "open_accounts": open_accounts,
            "credit_utilization_ratio": credit_utilization_ratio,
            TARGET_COL: y.astype(int),
        }
    )

    # Inject missing values
    n_missing = int(config.missing_rate * config.n_rows * len(RAW_FEATURES))
    if n_missing > 0:
        for _ in range(n_missing):
            r = rng.integers(0, config.n_rows)
            c = rng.choice(RAW_FEATURES)
            df.loc[r, c] = np.nan

    # Inject outliers (mostly in income and debt)
    n_outliers = int(config.outlier_rate * config.n_rows)
    if n_outliers > 0:
        outlier_rows = rng.choice(df.index.to_numpy(), size=n_outliers, replace=False)
        df.loc[outlier_rows, "income"] *= rng.uniform(4, 12, size=n_outliers)
        df.loc[outlier_rows, "debt_amount"] *= rng.uniform(3, 10, size=n_outliers)
        df.loc[outlier_rows, "late_payments_count"] += rng.integers(20, 60, size=n_outliers)
        df.loc[outlier_rows, "credit_utilization_ratio"] = np.clip(
            df.loc[outlier_rows, "credit_utilization_ratio"] + rng.uniform(0.5, 1.0, size=n_outliers), 0, 1
        )

    return df


def ensure_dataset_csv(path: Path, config: SyntheticDataConfig | None = None) -> Path:
    """
    Ensure a dataset exists at `path`. If missing, generate a synthetic CSV.
    Returns the path to the dataset CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    cfg = config or SyntheticDataConfig()
    df = generate_synthetic_credit_data(cfg)
    df.to_csv(path, index=False)
    return path

