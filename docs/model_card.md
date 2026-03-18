# Model Card — ChurnOps Multi-Domain Churn Prediction

## Model Overview

| Field | Details |
|-------|---------|
| **Model Name** | ChurnOps Multi-Domain Classifier |
| **Version** | 1.0.0 |
| **Type** | Binary Classification (Churn / Not Churned) |
| **Algorithms** | RandomForest, XGBoost, LightGBM, CatBoost, LogisticRegression, MLP |
| **Framework** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Training Data** | 8M synthetic rows across 8 domains |

## Intended Use

- **Primary Use**: Predict customer churn (attrition) probability across 8 industry domains.
- **Users**: Business analysts, marketing teams, data scientists evaluating MLOps architectures.
- **Out of Scope**: This model should not be used as the sole decision-maker for customer retention actions without human review.

## Supported Domains

| Domain | Features | Key Churn Drivers |
|--------|----------|-------------------|
| 📱 Telecom | Tenure, Contract, MonthlyCharges, Services | Short tenure + Month-to-month contract |
| 🏦 Banking | CreditScore, Balance, Transactions, Products | Low transaction frequency + single product |
| 🛒 E-commerce | OrderFrequency, Returns, AppActivity | High days since last purchase + high returns |
| 🎮 Gaming | WinRate, DailyActiveMinutes, Level, Purchases | Low daily activity + no purchases |
| 🎬 OTT/Streaming | WatchHours, LoginFrequency, PaymentFailures | Payment failures + low engagement |
| 🏥 Healthcare | Age, HealthScore, Claims, PremiumRegularity | Irregular premium payments + high claims |
| ☁️ SaaS | LoginFrequency, FeaturesUsed, SupportTickets | Low feature adoption + monthly billing |
| ✈️ Hospitality | BookingFrequency, LoyaltyPoints, Ratings | Low loyalty points + complaints |

## Performance Metrics (Telco Domain — Representative)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.88–0.92 |
| F1 Score | 0.78–0.83 |
| Accuracy | 0.82–0.86 |
| Precision | 0.80–0.85 |
| Recall | 0.75–0.82 |

> **Note**: Metrics vary by domain and training run. The ranges above represent typical performance across multiple training iterations.

## Decision Threshold

- Default threshold: **0.5** (churn probability ≥ 50% → classified as "Churned")
- For high-recall use cases (e.g., retention campaigns), consider lowering to **0.35–0.40** to catch more at-risk customers

## Training Data Details

- **Source**: Synthetically generated using NumPy distributions calibrated to mimic real-world patterns
- **Size**: ~1M rows per domain (8M total)
- **Class Balance**: ~26% churn / 74% retained (SMOTE applied during training)
- **Justification**: See [DATA_GENERATION_EXPLAINED.md](DATA_GENERATION_EXPLAINED.md) for detailed rationale

## Known Limitations & Biases

1. **Synthetic Data**: Models are trained on generated data, not real customer records. Performance on real-world data may differ.
2. **Feature Drift**: No continuous monitoring in the demo deployment — models do not automatically retrain on new data.
3. **Domain Simplification**: Each domain uses a reduced feature set compared to what a production system would employ.
4. **Geographic Bias**: Synthetic data does not account for regional/cultural differences in churn behavior.

## Ethical Considerations

- Churn predictions should supplement, not replace, human decision-making in customer retention strategies.
- Avoid using churn predictions to discriminate against customers in service quality or pricing.
- Customer data privacy must be maintained — the demo uses synthetic data specifically to avoid privacy concerns.

## How to Reproduce

```bash
# Generate synthetic data
python scripts/generate_8_domains.py

# Train models with MLflow tracking
python scripts/run_training.py --domain telco

# View results
mlflow ui
```
