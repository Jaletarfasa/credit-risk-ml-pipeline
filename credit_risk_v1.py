"""
credit_risk_v1.py

V1: Simple, reliable credit-risk pipeline for the MGB dataset.

- Loads C-level project data (Design II data.csv).
- Creates a binary default label (TARGET_DELQ).
- Does basic preprocessing:
    * Median imputation for numeric features
    * Most-frequent imputation + OneHotEncoding for categorical features
- Trains a RandomForestClassifier (base model).
- Handles class imbalance only by class_weight='balanced' (no SMOTE yet).
- Computes ROC AUC, PR AUC, and best F1 for DELQ over simple thresholds.

This is a clean teaching baseline we can build on for V2, V3, V4.
"""

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def run_credit_risk_v1(
    data_path: str,
    target_col: str = "ACCT_STATUS",
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Run the V1 MGB credit-risk pipeline.

    Parameters
    ----------
    data_path : str
        Path to Design II data.csv.
    target_col : str
        Name of the target column (ACCT_STATUS by default).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : sklearn Pipeline
        Fitted pipeline (preprocessing + RandomForest).
    dashboard : dict
        Contains:
            - metrics
            - class_distribution
            - threshold_reports
    """

    # ---------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------
    df = pd.read_csv(data_path, encoding="latin1")

    # ---------------------------------------------------
    # 2. Create binary target: TARGET_DELQ (1=DELQ, 0=non-DELQ)
    # ---------------------------------------------------
    if target_col == "ACCT_STATUS":
        df["TARGET_DELQ"] = (df["ACCT_STATUS"] == "DELQ").astype(int)
        target_binary_col = "TARGET_DELQ"
    else:
        target_binary_col = target_col  # assume already binary

    # Drop rows where target is missing
    df = df.dropna(subset=[target_binary_col])

    # ---------------------------------------------------
    # 3. Define feature sets (using raw variables only in V1)
    # ---------------------------------------------------
    numeric_features: List[str] = [
        "#_OF_MGB_ACCTS",
        "CURR_MGB_SAV_BAL",
        "AVG_MGB_CHQ_BAL",
        "#_OF_MGB_MSSD_PAY",
        "#_OF_DEPENDENTS",
        "AGE_VAR",
        "CREDIT_SCORE",
        "ANNUL_GROS_INC",
        "LOAN_VAL",
    ]

    categorical_features: List[str] = [
        "SEX_CD",
        "MARITAL_STATUS_CD",
        "LOAN_TYP_SHORT_DESC",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
    ]

    # Keep only rows with all feature columns present (some may still have NaNs, we impute)
    feature_cols = numeric_features + categorical_features
    df_model = df[feature_cols + [target_binary_col]].copy()

    # ---------------------------------------------------
    # 4. Inspect class imbalance (teaching point)
    # ---------------------------------------------------
    class_counts = df_model[target_binary_col].value_counts().sort_index()
    class_proportions = (class_counts / class_counts.sum()).round(4)

    class_distribution = {
        "counts": class_counts.to_dict(),
        "proportions": class_proportions.to_dict(),
        "imbalance_ratio": float(class_counts.max() / class_counts.min()),
    }

    print("Class distribution (TARGET_DELQ=0 non-default, 1 default):")
    print(class_counts)
    print("Proportions:", class_proportions.to_dict())
    print("Imbalance ratio (majority/minority):", class_distribution["imbalance_ratio"])

    # ---------------------------------------------------
    # 5. Train/test split
    # ---------------------------------------------------
    X = df_model[feature_cols]
    y = df_model[target_binary_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # ---------------------------------------------------
    # 6. Preprocessing pipeline
    # ---------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ---------------------------------------------------
    # 7. Random Forest base model
    #    class_weight='balanced' to partially address imbalance
    # ---------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf),
    ])

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # 8. Probabilities and metrics
    # ---------------------------------------------------
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print(f"ROC AUC (RandomForest, V1): {roc:.4f}")
    print(f"PR  AUC (RandomForest, V1): {pr:.4f}")

    # ---------------------------------------------------
    # 9. Threshold sweep for DELQ F1
    # ---------------------------------------------------
    thresholds = np.linspace(0.05, 0.5, 10)
    threshold_reports = []

    best_f1 = -1.0
    best_thr = None

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        report_dict = classification_report(
            y_test, y_pred_thr, output_dict=True, zero_division=0
        )
        f1_delq = report_dict["1"]["f1-score"]
        threshold_reports.append(
            {
                "threshold": float(thr),
                "f1_DELQ": float(f1_delq),
                "precision_DELQ": float(report_dict["1"]["precision"]),
                "recall_DELQ": float(report_dict["1"]["recall"]),
            }
        )
        if f1_delq > best_f1:
            best_f1 = f1_delq
            best_thr = thr

    print(f"Best threshold for class 1 (DELQ) F1: {best_thr}")
    print(f"Best F1 for DELQ (RandomForest V1): {best_f1}")

    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "best_threshold_f1_DELQ": float(best_thr),
        "best_f1_DELQ": float(best_f1),
    }

    dashboard: Dict[str, Any] = {
        "metrics": metrics,
        "class_distribution": class_distribution,
        "threshold_reports": threshold_reports,
    }

    return model, dashboard
