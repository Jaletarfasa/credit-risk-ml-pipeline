"""
credit_risk_v5.py

Final working credit-risk pipeline for the MGB / Design II dataset.

- Loads 'Design II data.csv'
- Builds binary target TARGET_DELQ from ACCT_STATUS
- Engineers ratio + tenure + age_bucket features
- Validates data before & after FE
- Builds preprocessing pipeline (impute + scale + one-hot)
- Handles imbalance with SMOTE + XGBoost(scale_pos_weight)
- Evaluates ROC AUC, PR AUC
- Searches best threshold for DELQ F1 between 0.05 and 0.50
- Returns (model, dashboard)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# --------------------------------------------------
# Small helper: data validation report
# --------------------------------------------------
def data_validation_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
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


# --------------------------------------------------
# Main function: run_credit_risk_v5
# --------------------------------------------------
def run_credit_risk_v5(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Final V5 credit-risk pipeline.

    Args:
        data_path: path to 'Design II data.csv'
        test_size: test split fraction
        random_state: seed for reproducibility

    Returns:
        model: fitted ImbPipeline (preprocess + SMOTE + XGBoost)
        dashboard: dict with metrics, validation tables, thresholds, etc.
    """

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv(data_path, encoding="latin1")

    # -----------------------------
    # 2. Create binary target
    # -----------------------------
    # ACCT_STATUS: GOOD / EARL / DELQ
    df["TARGET_DELQ"] = (df["ACCT_STATUS"] == "DELQ").astype(int)
    target_col = "TARGET_DELQ"

    # Class counts
    class_counts = df[target_col].value_counts().rename("count")

    # -----------------------------
    # 3. Basic feature engineering
    # -----------------------------
    # Convert dates
    df["REG_DATE"] = pd.to_datetime(df["REG_DATE"], errors="coerce")
    df["ENTRY_DT"] = pd.to_datetime(df["ENTRY_DT"], errors="coerce")

    # Tenure in years
    df["TENURE_DAYS"] = (df["ENTRY_DT"] - df["REG_DATE"]).dt.days
    df["TENURE_DAYS"] = df["TENURE_DAYS"].astype("float")
    df["customer_tenure_years"] = df["TENURE_DAYS"] / 365.25

    # Ratio features (with safe denominators)
    df["missed_pay_ratio"] = df["#_OF_MGB_MSSD_PAY"] / (df["#_OF_MGB_ACCTS"] + 1)
    df["savings_to_loan"] = df["CURR_MGB_SAV_BAL"] / (df["LOAN_VAL"] + 1)
    df["chq_to_income"] = df["AVG_MGB_CHQ_BAL"] / (df["ANNUL_GROS_INC"] + 1)
    df["loan_to_income"] = df["LOAN_VAL"] / (df["ANNUL_GROS_INC"] + 1)

    # Age buckets
    df["age_bucket"] = pd.cut(
        df["AGE_VAR"],
        bins=[19, 29, 39, 49, 59],
        labels=["20s", "30s", "40s", "50s"],
        include_lowest=True,
    )

    # -----------------------------
    # 4. Validation BEFORE modeling
    # -----------------------------
    validation_raw = data_validation_report(df, label="RAW_V5")

    # -----------------------------
    # 5. Define feature lists
    # -----------------------------
    numeric_features = [
        "#_OF_MGB_ACCTS",
        "CURR_MGB_SAV_BAL",
        "AVG_MGB_CHQ_BAL",
        "#_OF_MGB_MSSD_PAY",
        "#_OF_DEPENDENTS",
        "AGE_VAR",
        "CREDIT_SCORE",
        "ANNUL_GROS_INC",
        "LOAN_VAL",
        "TENURE_DAYS",
        "customer_tenure_years",
        "missed_pay_ratio",
        "savings_to_loan",
        "chq_to_income",
        "loan_to_income",
    ]

    categorical_features = [
        "SEX_CD",
        "MARITAL_STATUS_CD",
        "LOAN_TYP_SHORT_DESC",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "age_bucket",
    ]

    # Keep only rows with non-missing target
    df = df.dropna(subset=[target_col])

    # -----------------------------
    # 6. Validation AFTER FE
    # -----------------------------
    validation_fe = data_validation_report(df, label="AFTER_FEATURE_ENGINEERING_V5")

    # -----------------------------
    # 7. Preprocessing pipeline
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
    # 8. Train/test split
    # -----------------------------
    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # -----------------------------
    # 9. Handle imbalance + model
    # -----------------------------
    imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    print("Imbalance Ratio (majority/minority):", imbalance_ratio)

    clf = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        eval_metric="auc",
        random_state=random_state,
    )

    smote = SMOTE(random_state=random_state)

    model = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("balance", smote),
        ("model", clf),
    ])

    model.fit(X_train, y_train)

    # -----------------------------
    # 10. Base metrics
    # -----------------------------
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print("ROC AUC:", roc)
    print("PR AUC :", pr)

    # -----------------------------
    # 11. Threshold sweep for DELQ F1
    # -----------------------------
    thresholds = np.arange(0.05, 0.55, 0.05)
    records = []

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred_thr,
            labels=[1],
            zero_division=0,
        )
        records.append({
            "threshold": float(thr),
            "f1_DELQ": float(f1[0]),
            "precision_DELQ": float(prec[0]),
            "recall_DELQ": float(rec[0]),
        })

    threshold_table = pd.DataFrame(records)
    best_idx = threshold_table["f1_DELQ"].idxmax()
    best_row = threshold_table.loc[best_idx]

    print("Best threshold for class 1 F1:", best_row["threshold"])
    print("Best F1 for DELQ:", best_row["f1_DELQ"])

    # -----------------------------
    # 12. Build dashboard dict
    # -----------------------------
    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "best_threshold": float(best_row["threshold"]),
        "best_f1_DELQ": float(best_row["f1_DELQ"]),
        "best_precision_DELQ": float(best_row["precision_DELQ"]),
        "best_recall_DELQ": float(best_row["recall_DELQ"]),
        "imbalance_ratio": float(imbalance_ratio),
    }

    dashboard = {
        "metrics": metrics,
        "class_counts": class_counts,
        "validation_raw": validation_raw,
        "validation_fe": validation_fe,
        "threshold_table": threshold_table,
        "best_threshold_row": best_row,
    }

    return model, dashboard

if __name__ == "__main__":
    import joblib

    # 1) Train the model using your Design II dataset
    model, dashboard = run_credit_risk_v5(
        data_path=r"C:\Users\solom\Desktop\Design II data.csv"
        # or just "Design II data.csv" if the CSV is in the same folder as the script
    )

    # 2) Save the trained model for FastAPI / DVC
    joblib.dump(model, "best_credit_model.joblib")
    print("✅ Saved best_credit_model.joblib")

    # 3) (Optional) Save key metrics for inspection
    metrics_df = dashboard["threshold_table"]
    metrics_df.to_csv("threshold_metrics_v5.csv", index=False)
    print("✅ Saved threshold_metrics_v5.csv")
