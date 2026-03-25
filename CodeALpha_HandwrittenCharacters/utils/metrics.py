from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0)
    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": report}


