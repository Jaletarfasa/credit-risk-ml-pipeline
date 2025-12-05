"""
credit_risk_shap.py

Interpretability layer for the Design II credit risk model.

- Loads Design II data
- Re-applies the SAME feature engineering as credit_risk_v5.py
- Loads the trained ImbPipeline model from 'best_credit_model.joblib'
- Extracts the underlying XGBoost model
- Computes SHAP values on a sample of rows (post-preprocessing)
- Saves:
    - shap_summary.png         (global feature importance – beeswarm)
    - shap_summary_bar.png     (global feature importance – bar)
    - shap_decision_example.png (decision plot for one example)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
DATA_PATH = "Design II data.csv"         # same as credit_risk_v5.py
MODEL_PATH = "best_credit_model.joblib"  # trained ImbPipeline


# --------------------------------------------------
# 2. Load data (same file as in credit_risk_v5.py)
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="latin1")

# --------------------------------------------------
# 3. Create binary target (same as V5)
# --------------------------------------------------
df["TARGET_DELQ"] = (df["ACCT_STATUS"] == "DELQ").astype(int)
target_col = "TARGET_DELQ"

# --------------------------------------------------
# 4. REPEAT THE SAME FEATURE ENGINEERING AS V5
# --------------------------------------------------

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

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# --------------------------------------------------
# 5. Define feature lists (same as V5)
# --------------------------------------------------
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

all_features = numeric_features + categorical_features

# Build raw feature matrix
X = df[all_features].copy()
y = df[target_col].astype(int)

# For SHAP we can take a reasonably sized sample
sample_size = min(1000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

# --------------------------------------------------
# 6. Load trained model pipeline
# --------------------------------------------------
print("🔁 Loading trained ImbPipeline model from best_credit_model.joblib ...")
model = joblib.load(MODEL_PATH)

# model is ImbPipeline(preprocess -> SMOTE -> XGBClassifier)
preprocessor = model.named_steps["preprocess"]
xgb_model = model.named_steps["model"]

# --------------------------------------------------
# 7. Apply preprocessing (same as during training)
#    We transform X_sample into what the XGB model actually sees.
# --------------------------------------------------
X_transformed = preprocessor.transform(X_sample)

# Get feature names after preprocessing (numeric + one-hot encoded cats)
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]

# Ensure we have a dense array & wrap in DataFrame to keep names
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# --------------------------------------------------
# 8. Compute SHAP values with TreeExplainer
# --------------------------------------------------
print("📈 Fitting SHAP TreeExplainer on XGBoost model...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_transformed_df)

# shap_values is a shap.Explanation
values = shap_values.values  # shape: (n_samples, n_features)

# --------------------------------------------------
# 9. Save SHAP summary plots
# --------------------------------------------------

# Global summary (beeswarm)
plt.figure()
shap.summary_plot(
    values,
    X_transformed_df,
    feature_names=feature_names,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved shap_summary.png")

# Global summary bar
plt.figure()
shap.summary_plot(
    values,
    X_transformed_df,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved shap_summary_bar.png")

# --------------------------------------------------
# 10. SHAP decision plot for a single example
# --------------------------------------------------
idx = 0  # first sampled row

base_value = explainer.expected_value
# For binary classification this might be an array; take first element
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0]

shap.decision_plot(
    base_value,
    values[idx],            # 1D array for that example
    feature_names=feature_names,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_decision_example.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved shap_decision_example.png")

print("🎯 SHAP interpretability layer completed successfully.")
