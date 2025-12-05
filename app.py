from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import csv
import json
import time
from datetime import datetime


# ============================================================
# 1. Pydantic schema: INPUT FORMAT FOR THE API
#    (human-friendly names from your earlier design)
# ============================================================
class InputData(BaseModel):
    # core numeric inputs
    num_mgb_accts: int
    curr_mgb_sav_bal: float
    avg_mgb_chq_bal: float
    num_mgb_mssd_pay: int
    num_dependents: int
    age_var: int
    credit_score: float
    annual_gross_inc: float
    loan_val: float

    # categorical flags
    sex_cd: str               # e.g. "M", "F"
    marital_status_cd: str    # e.g. "S", "M"
    loan_typ_short_desc: str  # e.g. "AUTO", "MORTGAGE"
    flag_own_car: str         # e.g. "Y"/"N"
    flag_own_realty: str      # e.g. "Y"/"N"

    # optional dates (for tenure calculations)
    reg_date: Optional[str] = None   # "YYYY-MM-DD"
    entry_dt: Optional[str] = None   # "YYYY-MM-DD"


# ============================================================
# 2. SAME FEATURE ENGINEERING AS credit_risk_v5.py
# ============================================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the SAME feature engineering used in credit_risk_v5.py
    so that the FastAPI input matches the training pipeline.
    """

    # --- date conversion ---
    if "REG_DATE" in df.columns:
        df["REG_DATE"] = pd.to_datetime(df["REG_DATE"], errors="coerce")
    if "ENTRY_DT" in df.columns:
        df["ENTRY_DT"] = pd.to_datetime(df["ENTRY_DT"], errors="coerce")

    # Tenure features
    if "REG_DATE" in df.columns and "ENTRY_DT" in df.columns:
        df["TENURE_DAYS"] = (df["ENTRY_DT"] - df["REG_DATE"]).dt.days.astype("float")
        df["customer_tenure_years"] = df["TENURE_DAYS"] / 365.25
    else:
        df["TENURE_DAYS"] = 0.0
        df["customer_tenure_years"] = 0.0

    # Safe denominators for ratios
    if "#_OF_MGB_MSSD_PAY" in df.columns and "#_OF_MGB_ACCTS" in df.columns:
        df["missed_pay_ratio"] = df["#_OF_MGB_MSSD_PAY"] / (df["#_OF_MGB_ACCTS"] + 1)
    else:
        df["missed_pay_ratio"] = 0.0

    if "CURR_MGB_SAV_BAL" in df.columns and "LOAN_VAL" in df.columns:
        df["savings_to_loan"] = df["CURR_MGB_SAV_BAL"] / (df["LOAN_VAL"] + 1)
    else:
        df["savings_to_loan"] = 0.0

    if "AVG_MGB_CHQ_BAL" in df.columns and "ANNUL_GROS_INC" in df.columns:
        df["chq_to_income"] = df["AVG_MGB_CHQ_BAL"] / (df["ANNUL_GROS_INC"] + 1)
    else:
        df["chq_to_income"] = 0.0

    if "LOAN_VAL" in df.columns and "ANNUL_GROS_INC" in df.columns:
        df["loan_to_income"] = df["LOAN_VAL"] / (df["ANNUL_GROS_INC"] + 1)
    else:
        df["loan_to_income"] = 0.0

    # Age bucket
    if "AGE_VAR" in df.columns:
        df["age_bucket"] = pd.cut(
            df["AGE_VAR"],
            bins=[19, 29, 39, 49, 59],
            labels=["20s", "30s", "40s", "50s"],
            include_lowest=True,
        )
    else:
        df["age_bucket"] = "unknown"

    return df


# ============================================================
# 3. FEATURE LISTS — MUST MATCH credit_risk_v5.py
# ============================================================
NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "SEX_CD",
    "MARITAL_STATUS_CD",
    "LOAN_TYP_SHORT_DESC",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "age_bucket",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 4. LOAD MODEL + BEST THRESHOLD
# ============================================================
best_model = joblib.load("best_credit_model.joblib")

# Load best_threshold from metrics.json if available, else default 0.5
BEST_THRESHOLD_DEFAULT = 0.5
try:
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    best_threshold = float(metrics.get("best_threshold", BEST_THRESHOLD_DEFAULT))
except FileNotFoundError:
    best_threshold = BEST_THRESHOLD_DEFAULT


# ============================================================
# 5. LOGGING FUNCTION
# ============================================================
LOG_FILE = "credit_predictions_log.csv"


def log_prediction(
    payload: Dict[str, Any],
    proba_default: float,
    pred_class: int,
    elapsed_ms: Optional[float] = None,
    log_file: str = LOG_FILE,
) -> None:
    """
    Append a single prediction event to a CSV log file.
    Columns:
      - timestamp
      - all input fields from payload
      - default_probability
      - predicted_default
      - latency_ms
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    row: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "default_probability": float(proba_default),
        "predicted_default": int(pred_class),
        "latency_ms": float(elapsed_ms) if elapsed_ms is not None else None,
    }

    # include the original request fields
    for k, v in payload.items():
        row[k] = v

    fieldnames = list(row.keys())
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 6. FASTAPI APP + /predict ENDPOINT
# ============================================================
app = FastAPI(title="Credit Risk API (Design II + XGBoost + SHAP-ready)")


@app.post("/predict")
def predict_default(payload: InputData):
    start_time = time.time()

    # 1) Convert InputData (API schema) -> training column names
    p = payload.dict()

    row_for_training = {
        "#_OF_MGB_ACCTS": p["num_mgb_accts"],
        "CURR_MGB_SAV_BAL": p["curr_mgb_sav_bal"],
        "AVG_MGB_CHQ_BAL": p["avg_mgb_chq_bal"],
        "#_OF_MGB_MSSD_PAY": p["num_mgb_mssd_pay"],
        "#_OF_DEPENDENTS": p["num_dependents"],
        "AGE_VAR": p["age_var"],
        "CREDIT_SCORE": p["credit_score"],
        "ANNUL_GROS_INC": p["annual_gross_inc"],
        "LOAN_VAL": p["loan_val"],
        "SEX_CD": p["sex_cd"],
        "MARITAL_STATUS_CD": p["marital_status_cd"],
        "LOAN_TYP_SHORT_DESC": p["loan_typ_short_desc"],
        "FLAG_OWN_CAR": p["flag_own_car"],
        "FLAG_OWN_REALTY": p["flag_own_realty"],
        "REG_DATE": p.get("reg_date"),
        "ENTRY_DT": p.get("entry_dt"),
    }

    df_input = pd.DataFrame([row_for_training])

    # 2) Apply SAME feature engineering as training
    df_input = apply_feature_engineering(df_input)

    # 3) Select EXACT model features (avoid "columns are missing")
    X_model = df_input[ALL_MODEL_FEATURES].copy()

    # 4) Predict probability using trained ImbPipeline
    proba_default = best_model.predict_proba(X_model)[0, 1]
    predicted_default = int(proba_default >= best_threshold)

    elapsed_ms = (time.time() - start_time) * 1000.0

    # 5) Log the prediction event
    log_prediction(
        payload=p,
        proba_default=proba_default,
        pred_class=predicted_default,
        elapsed_ms=elapsed_ms,
    )

    return {
        "default_probability": float(proba_default),
        "predicted_default": predicted_default,
        "best_threshold": best_threshold,
        "latency_ms": round(elapsed_ms, 2),
    }


# ============================================================
# 7. LOCAL DEV ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
