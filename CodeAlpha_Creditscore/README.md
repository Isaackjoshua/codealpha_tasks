# Credit Scoring System (Python)

This project trains a **credit scoring classifier** to predict whether an individual is **creditworthy** (`good`) or a **bad credit risk** (`bad`) using structured financial + behavioral features.

It includes:
- Dataset loading + exploration (CSV)
- Synthetic dataset generation (if no CSV exists)
- Preprocessing: missing value imputation, encoding, scaling, and outlier handling
- Feature engineering (domain-inspired)
- Model training + comparison (Logistic Regression, Decision Tree, Random Forest) with cross-validation
- Evaluation metrics + visualizations (ROC, confusion matrices, feature importance)
- Model persistence + inference utility

## Project Structure

```
data/
models/
notebooks/        # optional
reports/
  figures/
utils/
train.py
evaluate.py
inference.py
requirements.txt
README.md
```

## Dataset

Expected columns (CSV):
- `income`
- `age`
- `employment_status`
- `debt_amount`
- `credit_history_length_years`
- `on_time_payment_ratio`
- `late_payments_count`
- `open_accounts`
- `credit_utilization_ratio`
- `creditworthy` (target: 1=good, 0=bad)

If `data/credit_data.csv` does not exist, running training will automatically generate a **synthetic dataset** for demonstration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Artifacts:
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/best_model.joblib`
- `models/training_metadata.json`
- `data/splits.joblib`

## Evaluate

```bash
python evaluate.py
```

Outputs:
- `reports/metrics_test.json`
- `reports/figures/roc_curves.png`
- `reports/figures/confusion_matrix_*.png`
- `reports/figures/feature_importance_*.png` (tree models)

## Inference

Interactive:
```bash
python inference.py
```

JSON payload:
```bash
python inference.py --json '{"income":55000,"age":34,"employment_status":"employed","debt_amount":12000,"credit_history_length_years":8,"on_time_payment_ratio":0.92,"late_payments_count":1,"open_accounts":6,"credit_utilization_ratio":0.28}'
```

The output includes a label (`good`/`bad`) and the predicted probability of being creditworthy.

## Streamlit UI

Run the web UI:

```bash
streamlit run streamlit_app.py
```

The UI supports:
- Training (button to run `train.py`)
- Single-applicant prediction (form inputs)
- Batch scoring from uploaded CSV
- Quick dataset preview (if `data/credit_data.csv` exists)

## Ethics & Bias (Important)

Credit scoring models can **amplify existing inequities** if trained on biased historical data or if features serve as proxies for protected attributes.

### Potential Bias Sources
- **Income disparity**: Lower income applicants may be unfairly penalized even when repayment behavior is strong.
- **Employment status**: Non-traditional employment (gig workers, self-employed) can be mischaracterized as higher risk.
- **Historical credit access**: People with limited credit history may be systematically scored lower (thin-file problem).
- **Proxy variables**: Even if protected attributes (e.g., race, ethnicity) are excluded, other variables may correlate with them.

### Fairness Concerns
- **Disparate impact**: Different approval rates across groups (even without explicit group labels).
- **Calibration differences**: Predicted probabilities may be well-calibrated for one group but not another.
- **Feedback loops**: Denials reduce future credit history, reinforcing disadvantage.

### Mitigation Strategies
- **Measure fairness**: Track metrics like demographic parity, equal opportunity, and calibration by group (when group labels are legally and ethically available).
- **Constrain or regularize**: Use simpler, interpretable models where appropriate; apply monotonic constraints in tree models (where supported) to enforce domain logic.
- **Reject inference on sensitive/proxy features**: Audit features for proxy risk; consider removing or coarsening high-risk proxies.
- **Human oversight**: Use ML as decision support, not fully automated decisions; provide appeal and recourse.
- **Robust validation**: Evaluate across time, segments, and economic conditions to avoid hidden brittleness.

This repository uses a synthetic dataset by default, but the same concerns apply to real deployments.
