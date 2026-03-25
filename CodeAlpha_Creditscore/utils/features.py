from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds domain-inspired features:
    - debt_to_income_ratio
    - payment_reliability_score
    - credit_utilization_pct
    """

    def fit(self, x: pd.DataFrame, y=None):  # noqa: ANN001
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()

        income = x.get("income")
        debt = x.get("debt_amount")
        if income is not None and debt is not None:
            x["debt_to_income_ratio"] = debt / np.maximum(income, 1.0)

        on_time = x.get("on_time_payment_ratio")
        late = x.get("late_payments_count")
        if on_time is not None and late is not None:
            # Higher is better. Penalize late payments more when on-time ratio is low.
            penalty = np.log1p(np.maximum(late, 0)) / 5.0
            x["payment_reliability_score"] = np.clip(on_time - penalty, -1.0, 1.0)

        util = x.get("credit_utilization_ratio")
        if util is not None:
            x["credit_utilization_pct"] = util * 100.0

        return x


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: list[str] | None = None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, x: pd.DataFrame, y=None):  # noqa: ANN001
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        return x.drop(columns=[c for c in self.columns_to_drop if c in x.columns], errors="ignore")

