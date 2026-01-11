# 💳 TransactDetect 

A full pipeline for detecting fraudulent financial transactions using machine learning, addressing severe class imbalance on a real-world simulated dataset with over 6 million records. Built and deployed for practical use in real-time fraud monitoring systems.

---

## 📁 Dataset

- Source: [PaySim Mobile Transaction Simulation Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Records: 6,362,620 transactions
- Features include transaction type, amount, balance before/after, and fraud label.
- Target: `isFraud` (1 = fraudulent, 0 = genuine)

---

## 🧠 Problem Statement

Credit card fraud is rare but extremely costly. The challenge lies in detecting fraudulent activity from highly imbalanced data (~0.13% fraud cases) while minimizing false positives in genuine transactions.

---

## ⚙️ Key Components

- **Data Preprocessing**: 
  - Cleaned & transformed transactional data
  - Encoded transaction types
  - Created new features: balance deltas, transaction anomalies

- **Handling Imbalance**:
  - Explored SMOTE, RandomOverSampler, Class weighting
  - Final model tuned with `class_weight=balanced` and stratified sampling

- **Modeling**:
  - Baseline: Logistic Regression
  - Advanced: Random Forest, XGBoost with hyperparameter tuning (via `GridSearchCV`)
  - Evaluation metrics: F1-score, G-Mean, ROC-AUC, MCC

- **Performance**:
  - Achieved significant performance gain: +15% ROC-AUC over baseline
  - Confusion matrix analysis showed reduced false negatives

---

## 🧪 Evaluation Metrics

| Metric       | Value (XGBoost) |
|--------------|----------------|
| Accuracy     | ~99.9%         |
| ROC-AUC      | ~0.98+         |
| F1-Score     | ~0.82          |
| MCC          | ~0.83          |

---

## 📦 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, imbalanced-learn

Install with:

```bash
pip install -r requirements.txt
