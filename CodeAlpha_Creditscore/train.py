from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from utils.config import ProjectPaths, RANDOM_STATE, TARGET_COL
from utils.data import explore_dataset, load_dataset, split_features_target
from utils.features import FeatureEngineer
from utils.preprocess import build_preprocessor
from utils.synthetic_data import SyntheticDataConfig, ensure_dataset_csv


def _model_grids():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 0.5, 1.0, 2.0],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
    }


def build_pipeline(model):
    feature_engineer = FeatureEngineer()
    # After feature engineering, build a preprocessor based on known columns.
    engineered_cols = [
        "income",
        "age",
        "employment_status",
        "debt_amount",
        "credit_history_length_years",
        "on_time_payment_ratio",
        "late_payments_count",
        "open_accounts",
        "credit_utilization_ratio",
        "debt_to_income_ratio",
        "payment_reliability_score",
        "credit_utilization_pct",
    ]
    preprocessor, _ = build_preprocessor(engineered_cols)

    return Pipeline(
        steps=[
            ("features", feature_engineer),
            ("preprocess", preprocessor),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("model", model),
        ]
    )


def main() -> int:
    paths = ProjectPaths.from_cwd()

    parser = argparse.ArgumentParser(description="Train credit scoring models and select the best one.")
    parser.add_argument("--data", type=str, default=str(paths.data_dir / "credit_data.csv"), help="Path to CSV dataset.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split fraction (from remaining).")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for GridSearchCV.")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    ensure_dataset_csv(dataset_path, SyntheticDataConfig(n_rows=5000, random_state=args.seed))
    df = load_dataset(dataset_path)
    summary = explore_dataset(df)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Shape: {summary.shape}")
    print("\nMissing values (top 10):")
    print(summary.missing_by_column.head(10).to_string())
    if not summary.class_balance.empty:
        print("\nClass balance (proportion):")
        print(summary.class_balance.to_string())
    print("\nHead:")
    print(df.head(5).to_string(index=False))

    x, y = split_features_target(df)

    # Simple feature selection signal: correlation of numeric features with target (good=1).
    numeric_cols = [c for c in x.columns if c != "employment_status" and pd.api.types.is_numeric_dtype(x[c])]
    if numeric_cols:
        corr = (
            pd.concat([x[numeric_cols], y.rename(TARGET_COL)], axis=1)
            .corr(numeric_only=True)[TARGET_COL]
            .drop(labels=[TARGET_COL], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        print("\nNumeric feature correlations with target (|corr| descending):")
        print(corr.to_string())

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    # Validation is primarily for a final sanity check; model selection is based on CV ROC-AUC.
    val_fraction_of_trainval = args.val_size / max(1e-9, (1.0 - args.test_size))
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=args.seed,
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    results = []
    trained_models: dict[str, object] = {}

    for name, (model, grid) in _model_grids().items():
        pipe = build_pipeline(model)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            refit=True,
            verbose=0,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        trained_models[name] = best

        # Evaluate on validation for sanity check (not used for selection)
        val_proba = best.predict_proba(x_val)[:, 1]
        from sklearn.metrics import roc_auc_score

        val_auc = float(roc_auc_score(y_val, val_proba))

        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_best_roc_auc": float(search.best_score_),
                "val_roc_auc": val_auc,
            }
        )
        print(f"\n{name}: CV ROC-AUC={search.best_score_:.4f} | Val ROC-AUC={val_auc:.4f}")
        print(f"Best params: {search.best_params_}")

    results_df = pd.DataFrame(results).sort_values(by="cv_best_roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[str(best_name)]

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained_models.items():
        joblib.dump(model, paths.models_dir / f"{name}.joblib")

    joblib.dump(best_model, paths.models_dir / "best_model.joblib")

    metadata = {
        "dataset": str(dataset_path),
        "target": TARGET_COL,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "selection_metric": "roc_auc (cross-validation)",
        "ranking": results_df.to_dict(orient="records"),
        "best_model": str(best_name),
    }
    (paths.models_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Report test performance for the chosen model
    test_proba = best_model.predict_proba(x_test)[:, 1]
    from sklearn.metrics import roc_auc_score

    test_auc = float(roc_auc_score(y_test, test_proba))
    print("\nModel ranking (by CV ROC-AUC):")
    print(results_df[["model", "cv_best_roc_auc", "val_roc_auc"]].to_string(index=False))
    print(f"\nSelected best model: {best_name} | Test ROC-AUC={test_auc:.4f}")

    # Save split data for evaluate.py consistency
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    split_path = paths.data_dir / "splits.joblib"
    joblib.dump(
        {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        },
        split_path,
    )
    print(f"Saved models to: {paths.models_dir}")
    print(f"Saved splits to: {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
