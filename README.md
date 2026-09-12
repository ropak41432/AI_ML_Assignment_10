# Telco Customer Churn Prediction

Predicting customer churn using the Telco dataset across four classification models: Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest.

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 80.70% | 65.84% | 56.68% | 0.6092 | 0.8418 |
| **Random Forest** | 80.06% | 67.16% | 48.66% | 0.5643 | 0.8438 |
| **Decision Tree** | 79.56% | 63.35% | 54.55% | 0.5862 | 0.8270 |
| **K-Nearest Neighbors** | 77.36% | 57.70% | 55.08% | 0.5636 | 0.8068 |

## Key Findings
- **Best Model:** Random Forest achieved the highest ROC-AUC (0.8438), showing superior ranking ability across thresholds.
- **Metric Priority:** Recall is prioritized over Precision to minimize costly customer churn (False Negatives).
- **Churn Drivers:** Month-to-month contracts, low tenure (< 12 months), and high monthly charges are the primary factors leading to customer attrition.

## How to Run

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the pipeline:
```bash
py customer_churn_prediction.py
```
