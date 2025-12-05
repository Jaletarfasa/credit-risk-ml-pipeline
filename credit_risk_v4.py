"""
credit_risk_v4.py

Teaching-quality credit risk pipeline for the MGB dataset.

- V1: Baseline model (Random Forest, accuracy trap with imbalance)
- V2: Threshold tuning for DELQ (class 1) F1
- V3: Feature engineering + NaN / SMOTE issues revealed by validation
- V4: Robust preprocessing + safe ratios + data validation + XGBoost + SMOTE + threshold search

This file implements V4 directly:
- Loads 'Design II data.csv'
- Engineers behavioral & risk features
- Runs validation before & after FE
- Builds a scikit-learn + imbalanced-learn pipeline with XGBoost
- Searches best threshold for class 1 F1
- Returns metrics, validation summaries, and feature importance
"""

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# =========================================================
# 1. Data Quality Agent (rule-based validation)
# =========================================================

class DataQualityAgent:
    """
    Simple, burden-reducing validation agent.

    Summarises:
    - dtypes
    - missing counts / percentages
    - number of unique values
    - min / max (numerics)
    - constant / high-cardinality flags
    """

    def validate(self, df: pd.DataFrame, label: str = "DATA") -> pd.DataFrame:
        print("\n==============================")
        print(f"🔍 DATA VALIDATION: {label}")
        print("==============================")

        report = pd.DataFrame({
            "dtype": df.dtypes,
            "missing_cnt": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique()
        })

        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            report.loc[num_cols, "min"] = df[num_cols].min()
            report.loc[num_cols, "max"] = df[num_cols].max()

        report["constant_col"] = df.nunique() == 1
        report["high_cardinality"] = df.nunique() > (0.5 * len(df))

        return report


# =========================================================
# 2. Feature Engineering (V4, robust to NaNs)
# =========================================================

def engineer_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create robust behavioral and risk features:

    - missed_pay_ratio  = # missed / (# accounts + 1)
    - savings_to_loan   = savings / (loan + 1)
    - chq_to_income     = avg chequing / (income + 1)
    - loan_to_income    = loan / (income + 1)
    - customer_tenure_years from REG_DATE, ENTRY_DT
    - age_bucket categories from AGE_VAR
    """

    df = df.copy()

    # Ensure numeric type where needed (if read as object)
    for col in [
        "#_OF_MGB_MSSD_PAY",
        "#_OF_MGB_ACCTS",
        "CURR_MGB_SAV_BAL",
        "AVG_MGB_CHQ_BAL",
        "ANNUL_GROS_INC",
        "LOAN_VAL",
        "AGE_VAR",
        "CREDIT_SCORE",
        "#_OF_DEPENDENTS",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe ratio denominators (fillna(0), then add 1 to avoid division by zero)
    miss = df["#_OF_MGB_MSSD_PAY"].fillna(0)
    accts = df["#_OF_MGB_ACCTS"].fillna(0)
    savings = df["CURR_MGB_SAV_BAL"].fillna(0)
    chq = df["AVG_MGB_CHQ_BAL"].fillna(0)
    income = df["ANNUL_GROS_INC"].fillna(0)
    loan = df["LOAN_VAL"].fillna(0)

    df["missed_pay_ratio"] = miss / (accts + 1.0)
    df["savings_to_loan"] = savings / (loan + 1.0)
    df["chq_to_income"] = chq / (income + 1.0)
    df["loan_to_income"] = loan / (income + 1.0)

    # Tenure in years
    if "REG_DATE" in df.columns and "ENTRY_DT" in df.columns:
        df["REG_DATE"] = pd.to_datetime(df["REG_DATE"], errors="coerce")
        df["ENTRY_DT"] = pd.to_datetime(df["ENTRY_DT"], errors="coerce")
        tenure_days = (df["ENTRY_DT"] - df["REG_DATE"]).dt.days
        df["customer_tenure_years"] = tenure_days / 365.25
    else:
        df["customer_tenure_years"] = np.nan

    # Age bucket
    if "AGE_VAR" in df.columns:
        df["age_bucket"] = pd.cut(
            df["AGE_VAR"],
            bins=[19, 29, 39, 49, 59],
            labels=["20-29", "30-39", "40-49", "50-59"]
        )
    else:
        df["age_bucket"] = pd.Series(pd.Categorical([np.nan] * len(df)))

    return df


# =========================================================
# 3. Threshold Search Utility
# =========================================================

def find_best_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Search over thresholds and return:
    - best_threshold for class 1 F1
    - best_f1
    - dict of threshold -> F1

    y_true: binary (0/1)
    y_proba: predicted probabilities for class 1
    """

    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    f1_dict = {}
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f1_dict[thr] = f1

    # Select best
    best_thr = max(f1_dict, key=f1_dict.get)
    best_f1 = f1_dict[best_thr]
    return best_thr, best_f1, f1_dict


# =========================================================
# 4. Main Orchestrator: run_credit_risk_v4
# =========================================================

def run_credit_risk_v4(
    data_path: str,
    target_col: str = "ACCT_STATUS",
) -> Tuple[ImbPipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Run the full V4 credit-risk pipeline.

    Steps:
    - Load CSV
    - Engineer features
    - Run data validation (RAW + AFTER FE)
    - Build preprocessor (imputation + scaling + OHE)
    - XGBoost + SMOTE
    - Compute ROC AUC, PR AUC
    - Search best threshold for DELQ F1
    - Build feature importance table

    Returns:
        model        : fitted ImbPipeline
        dashboard    : dict with metrics, validations, feature_importances
        agents_out   : dict with DataQualityAgent instance (and any future agents)
    """

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_path, encoding="latin1")

    # -----------------------------
    # Agents
    # -----------------------------
    dq_agent = DataQualityAgent()

    # Validation: raw
    validation_raw = dq_agent.validate(df, label="RAW_V4")

    # -----------------------------
    # Feature engineering V4
    # -----------------------------
    df = engineer_features_v4(df)

    # Create binary target: 1 = DELQ, 0 = non-DELQ
    if target_col == "ACCT_STATUS":
        df["TARGET_DELQ"] = (df["ACCT_STATUS"] == "DELQ").astype(int)
        target_binary_col = "TARGET_DELQ"
    else:
        target_binary_col = target_col  # assume already binary

    # Drop rows with missing target
    df = df.dropna(subset=[target_binary_col])

    # Validation: after FE
    validation_fe = dq_agent.validate(df, label="AFTER_FEATURE_ENGINEERING_V4")

    # -----------------------------
    # Feature sets
    # -----------------------------
    numeric_features = [
        "#_OF_MGB_ACCTS",
        "#_OF_MGB_MSSD_PAY",
        "CURR_MGB_SAV_BAL",
        "AVG_MGB_CHQ_BAL",
        "#_OF_DEPENDENTS",
        "AGE_VAR",
        "CREDIT_SCORE",
        "ANNUL_GROS_INC",
        "LOAN_VAL",
        "missed_pay_ratio",
        "savings_to_loan",
        "chq_to_income",
        "loan_to_income",
        "customer_tenure_years",
    ]

    categorical_features = [
        "SEX_CD",
        "MARITAL_STATUS_CD",
        "LOAN_TYP_SHORT_DESC",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "age_bucket",
    ]

    # Restrict to columns that exist (defensive)
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # -----------------------------
    # Train/test split
    # -----------------------------
    y = df[target_binary_col].astype(int)
    X = df[numeric_features + categorical_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Imbalance ratio (majority/minority)
    counts = y_train.value_counts()
    maj = counts[0]
    mino = counts[1]
    imbalance_ratio = maj / mino
    print("Imbalance Ratio (majority/minority):", imbalance_ratio)

    # -----------------------------
    # Preprocessor
    # -----------------------------
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

    # -----------------------------
    # Model: XGBoost + SMOTE
    # -----------------------------
    clf = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,  # important for imbalance
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    smote = SMOTE(random_state=42)

    model = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("balance", smote),
        ("model", clf),
    ])

    model.fit(X_train, y_train)

    # -----------------------------
    # Metrics
    # -----------------------------
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    best_thr, best_f1, f1_dict = find_best_f1_threshold(y_test.values, y_proba)

    print("ROC AUC:", roc)
    print("PR AUC :", pr)
    print("Best threshold for class 1 F1:", best_thr)
    print("Best F1 for DELQ:", best_f1)

    # Classification report at best threshold
    y_pred_best = (y_proba >= best_thr).astype(int)
    cls_report_text = classification_report(y_test, y_pred_best)

    # -----------------------------
    # Feature importance (approximate)
    # -----------------------------
    # Extract feature names after preprocessing
    preprocess_fitted = model.named_steps["preprocess"]
    feature_names_transformed = preprocess_fitted.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names_transformed,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # -----------------------------
    # Dashboard & agents
    # -----------------------------
    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "best_threshold_f1": float(best_thr),
        "best_f1_delq": float(best_f1),
        "f1_by_threshold": f1_dict,
        "classification_report_best_thr": cls_report_text,
        "imbalance_ratio": float(imbalance_ratio),
    }

    dashboard = {
        "metrics": metrics,
        "validation_raw": validation_raw,
        "validation_fe": validation_fe,
        "feature_importances": fi_df,
    }

    agents_out = {
        "data_quality_agent": dq_agent,
    }

    return model, dashboard, agents_out

