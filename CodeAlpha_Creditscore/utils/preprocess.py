from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import RAW_FEATURES


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Clip numeric features using IQR-based bounds computed on the training split.
    Works on numpy arrays (after imputation) inside a Pipeline.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y=None):  # noqa: ANN001
        q1 = np.nanpercentile(x, 25, axis=0)
        q3 = np.nanpercentile(x, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("IQRCapper must be fit before transform.")
        return np.clip(x, self.lower_, self.upper_)


@dataclass(frozen=True)
class PreprocessSpec:
    numeric_features: List[str]
    categorical_features: List[str]


def infer_feature_types(df: pd.DataFrame) -> PreprocessSpec:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return PreprocessSpec(numeric_features=numeric, categorical_features=categorical)


def build_preprocessor(feature_columns: list[str]) -> tuple[ColumnTransformer, PreprocessSpec]:
    tmp = pd.DataFrame({c: pd.Series(dtype="float64") for c in feature_columns})
    if "employment_status" in tmp.columns:
        tmp["employment_status"] = pd.Series(dtype="object")
    spec = infer_feature_types(tmp)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outliers", IQRCapper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, spec

