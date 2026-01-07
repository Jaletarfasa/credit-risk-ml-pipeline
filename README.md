# 🚀 Credit Risk ML Pipeline (End-to-End MLOps System)

A production-style **credit-risk modeling system** showcasing end-to-end delivery: data preparation → model training → tracking → API serving → interpretability → monitoring, packaged with **engineering hygiene** (issues, PR workflow, validation checks).

## ✅ What this demonstrates
- Translating requirements into a working solution (model + API + monitoring)
- Shipping incrementally with **Git-based workflow** and review-ready code
- Implementing **basic quality gates** (formatting, validation checks, naming conventions)

---

## 🔍 Key Features
- ✅ Data cleaning, feature engineering, preprocessing  
- ✅ XGBoost credit-risk classifier with SMOTE for class imbalance  
- ✅ Experiment + artifact tracking using **DVC**  
- ✅ Prediction API using **FastAPI + Uvicorn**  
- ✅ Model interpretability using **SHAP** (global + local explanations)  
- ✅ Monitoring pipeline (distribution drift checks + prediction logs)  
- ✅ Git-based collaboration workflow (branch → commit → merge)

---

## 🔧 Tech Stack

| Layer | Tools |
|------|-------|
| Modeling | Python, Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn |
| MLOps | DVC (pipelines, artifacts, metrics), GitHub |
| API Deployment | FastAPI, Uvicorn |
| Interpretability | SHAP |
| Logging / Monitoring | DVC metrics + CSV logs |

---

## 📁 Repository Structure (Key Files)

| File | Purpose |
|------|---------|
| `credit_risk_v5.py` | Training pipeline (feature engineering → training → DVC logging) |
| `app.py` | FastAPI inference API with preprocessing + prediction logging |
| `credit_risk_shap.py` | SHAP interpretability plots (global + local) |
| `dvc.yaml` | DVC stages (train, evaluate, monitor) |
| `shap_dashboard_combined.png` | Explainability visual summary |
| `requirements.txt` | Python dependencies |

---

## 🧪 Quality Gates (Clean + Tested Delivery)

This project includes lightweight, practical checks to ensure reliable outputs:
- Input schema / null checks (required fields present)
- Basic range / type validation for numeric inputs
- Reproducible pipeline execution via **DVC**
- Consistent naming conventions and structured logging outputs

> These checks mirror “clean, tested code + quality gates” expectations in sprint-based teams.

---

## 🧠 Model Explainability (SHAP)

- **Global Importance:** which features drive risk the most  
- **Local Explanations:** why a specific customer gets a given risk score  
- **Decision Plots:** feature-by-feature reasoning  

This supports transparency expectations common in finance/regulatory settings.

---

## 🏃 How to Run Locally

### 1) Clone the repository
```bash
git clone https://github.com/Jaletarfasa/credit-risk-ml-pipeline.git
cd credit-risk-ml-pipeline
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
