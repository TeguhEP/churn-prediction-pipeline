# Predicting Customer Churn Before It Happens
## A Production-Ready ML Pipeline with Business-Driven Threshold Optimisation

**Domain:** Telecom · Customer Retention · Predictive Analytics  
**Tools:** Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn  
**Status:** Complete — fully documented end-to-end pipeline

---

## Business Problem

A telecom company loses revenue every month to customers who cancel their
subscriptions without any prior warning to the retention team. This project
builds a complete machine learning pipeline that converts raw customer data
into a risk-ranked contact list — identifying which customers are most likely
to churn before they actually cancel, giving the retention team a targeted,
data-driven outreach list.

---

## Key Results

| Metric | Result |
|--------|--------|
| Accuracy | 81.5% |
| Precision | 83.2% |
| Recall | 79.0% |
| F1-Score | 81.0% |
| ROC-AUC | 89.0% |
| Average Precision | 90.1% |
| CV Mean ROC-AUC | 87.6% ± 2.5% |

---

## Pipeline Overview
```
Raw Data (2,000 customers · 10 features)
         ↓
Exploratory Data Analysis (6 diagnostic visualisations)
         ↓
Stratified Train / Test Split — 80% / 20%
         ↓
Feature Scaling — StandardScaler
         ↓
Baseline Model — LogisticRegression (default parameters)
         ↓
5-Fold Stratified Cross-Validation (5 metrics)
         ↓
Hyperparameter Tuning — GridSearchCV + RandomizedSearchCV
         ↓
Final Tuned Model — LogisticRegression (C=1.0, L1, liblinear)
         ↓
Full Evaluation Suite (7 metrics)
         ↓
Business Deployment Analysis (threshold · calibration · sigmoid)
         ↓
Risk-Tiered Actionable Output
  → 151 High Risk   → Personal outreach call
  → 85  Medium Risk → Email campaign
  → 164 Low Risk    → No action
```

---

## Repository Structure
```
churn-prediction-pipeline/
├── notebooks/
│   └── churn_prediction_complete.ipynb   ← Full pipeline
├── src/
│   ├── data_generator.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── scorer.py
├── outputs/                              ← All saved charts
├── models/                               ← Serialized model files
├── requirements.txt
└── README.md
```

---

## Quick Start
```bash
# Clone the repository
git clone https://github.com/YOURUSERNAME/churn-prediction-pipeline.git
cd churn-prediction-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
jupyter notebook notebooks/churn_prediction_complete.ipynb
```

---

## Three Principal Findings

**1 — Late payment behaviour dominates all other predictors by a factor of two.**
The `late_payments` feature received a model coefficient of 2.352 — more than
twice the second-largest predictor. This single feature, observable in any CRM
system, is sufficient to trigger an immediate retention escalation rule
independently of the full model pipeline.

**2 — Hyperparameter tuning confirmed the baseline was already near-optimal.**
GridSearchCV across 36 configurations and RandomizedSearchCV across 60 samples
both produced improvements of less than 0.0003 ROC-AUC over the default
parameters — confirming a flat hyperparameter landscape and redirecting future
improvement effort toward feature engineering rather than parameter optimisation.

**3 — The classification threshold is a business cost decision, not a default.**
Systematic analysis across 300 threshold values identified that the default 0.5
threshold is financially suboptimal given the 21:1 to 108:1 cost asymmetry
between missed churners and wasted retention calls. The threshold analysis
framework provides the tool to select the cost-optimal operating point for any
given intervention economics.

---

## Full Report

The complete 15-part portfolio report including business context, methodology,
chart explanations, and deployment recommendations is available at:
[Full Report PDF Link](https://drive.google.com/file/d/1WgISq0Sl5ZxuDYUO7QlcsczPzLpN4plc/view?usp=drive_link)

---

## Contact

Teguh Eka Prahara · prahara.teguh@gmail.com · [My LinkedIn](https://www.linkedin.com/in/prahara89/)
