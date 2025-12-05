# credit-risk-ml-pipeline
# 🚀 Credit Risk ML Pipeline (End-to-End MLOps System)

This project is a **full production-grade credit-risk modeling system**, built to demonstrate real-world data science, ML engineering, and MLOps skills.

It includes:

- ✅ Data cleaning, feature engineering, and preprocessing  
- ✅ XGBoost credit-risk classifier with SMOTE for imbalance  
- ✅ Full MLOps tracking using **DVC**  
- ✅ Scalable prediction API using **FastAPI + Uvicorn**  
- ✅ Model interpretability using **SHAP global & local explanations**  
- ✅ Automated monitoring pipeline (distribution drift, prediction logs)  
- ✅ Git-based collaboration with clean versioning  

---

## 🔧 Tech Stack

| Layer | Tools |
|------|-------|
| **Modeling** | Python, Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn |
| **MLOps** | DVC (pipelines, artifacts), GitHub |
| **API Deployment** | FastAPI, Uvicorn |
| **Interpretability** | SHAP (summary, bar, decision plots) |
| **Logging / Monitoring** | DVC metrics + CSV logs |

---

## 📊 Key Files

| File | Purpose |
|------|---------|
| `credit_risk_v5.py` | Full training pipeline (feature engineering → training → DVC logging) |
| `app.py` | FastAPI prediction API with preprocessing + logging |
| `credit_risk_shap.py` | SHAP interpretability dashboard & plots |
| `dvc.yaml` | Defines ML stages (train, evaluate, monitor) |
| `shap_dashboard_combined.png` | Visual summary for explainability |
| `requirements.txt` | Environment dependencies |

---

## 🧠 SHAP Interpretability

This project includes:

- **Global Importance:** Which features matter most  
- **Local Explanations:** Why THIS customer got their risk score  
- **Decision Plots:** Feature-by-feature reasoning  

SHAP ensures regulatory transparency — essential for finance.

---

## 🔥 How to Run Locally

### 1. Clone repo

```bash
git clone https://github.com/Jaletarfasa/credit-risk-ml-pipeline.git
cd credit-risk-ml-pipeline
