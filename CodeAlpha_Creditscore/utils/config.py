from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    @staticmethod
    def from_cwd() -> "ProjectPaths":
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data"
        models_dir = root / "models"
        reports_dir = root / "reports"
        figures_dir = reports_dir / "figures"
        return ProjectPaths(
            root=root,
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
        )


RANDOM_STATE = 42
TARGET_COL = "creditworthy"

# Raw input columns (expected in CSV and in inference payloads)
RAW_FEATURES = [
    "income",
    "age",
    "employment_status",
    "debt_amount",
    "credit_history_length_years",
    "on_time_payment_ratio",
    "late_payments_count",
    "open_accounts",
    "credit_utilization_ratio",
]

