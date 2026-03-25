from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils.config import RAW_FEATURES, TARGET_COL


@dataclass(frozen=True)
class DatasetSummary:
    shape: Tuple[int, int]
    missing_by_column: pd.Series
    class_balance: pd.Series


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def explore_dataset(df: pd.DataFrame) -> DatasetSummary:
    missing = df.isna().sum().sort_values(ascending=False)
    if TARGET_COL in df.columns:
        class_balance = df[TARGET_COL].value_counts(normalize=True).rename("proportion")
    else:
        class_balance = pd.Series(dtype=float, name="proportion")
    return DatasetSummary(shape=df.shape, missing_by_column=missing, class_balance=class_balance)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_cols = [c for c in RAW_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")
    x = df[RAW_FEATURES].copy()
    y = df[TARGET_COL].astype(int).copy()
    return x, y

