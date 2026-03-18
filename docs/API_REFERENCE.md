# ChurnOps — API Reference

> Complete documentation for the ChurnOps FastAPI REST API.
> Base URL: `http://localhost:8000`

---

## Authentication

No authentication is required in the current version. All endpoints are public.

---

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check / liveness probe |
| `GET` | `/api/v1/model/info` | Model metadata and metrics |
| `POST` | `/api/v1/predict` | Single customer prediction |
| `POST` | `/api/v1/predict/batch` | Batch prediction (JSON array) |
| `POST` | `/api/v1/predict/upload` | Batch prediction (CSV file upload) |
| `POST` | `/api/v1/explain` | SHAP-based prediction explanation |
| `GET` | `/api/v1/monitoring/metrics` | Prometheus-compatible metrics |

---

## `GET /health`

Returns the API health status. Used by Docker and Kubernetes for liveness probes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-03-18T10:00:00"
}
```

---

## `POST /api/v1/predict`

Predict churn for a single customer.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `domain` | string | `telco` | Industry domain (`telco`, `banking`, `ecommerce`, `saas`, `healthcare`, `gaming`, `ott`, `hospitality`) |

**Request Body** (Telco example):
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 1397.5
}
```

**Response:**
```json
{
  "prediction": 1,
  "churn_probability": 0.8234,
  "label": "Churned",
  "confidence": 0.8234,
  "timestamp": "2026-03-18T10:00:00"
}
```

**Field Details:**
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | `0` = Not Churned, `1` = Churned |
| `churn_probability` | float | Probability of churn (0.0 – 1.0) |
| `label` | string | Human-readable label |
| `confidence` | float | Model confidence (= max(prob, 1-prob)) |

---

## `POST /api/v1/predict/batch`

Predict churn for multiple customers via a JSON array.

**Query Parameters:** Same as `/predict`.

**Request Body:**
```json
{
  "customers": [
    { "tenure": 12, "MonthlyCharges": 70.35, ... },
    { "tenure": 45, "MonthlyCharges": 42.30, ... }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "prediction": 1, "churn_probability": 0.82, "label": "Churned", "confidence": 0.82, "timestamp": "..." },
    { "prediction": 0, "churn_probability": 0.15, "label": "Not Churned", "confidence": 0.85, "timestamp": "..." }
  ],
  "total": 2,
  "churned_count": 1,
  "churn_rate": 0.5
}
```

---

## `POST /api/v1/predict/upload`

Upload a CSV file for batch predictions. Returns per-row results plus aggregate analytics.

**Query Parameters:** Same as `/predict`.

**Request:** Multipart form data with `file` field containing a `.csv` file.

```bash
curl -X POST "http://localhost:8000/api/v1/predict/upload?domain=telco" \
  -F "file=@customers.csv"
```

**Response:**
```json
{
  "total": 1000,
  "churned_count": 320,
  "churn_rate": 0.32,
  "avg_probability": 0.4125,
  "probability_distribution": [
    { "bin": "0-10%", "count": 150 },
    { "bin": "10-20%", "count": 120 },
    ...
  ],
  "confidence_breakdown": {
    "low": 580,
    "medium": 200,
    "high": 220
  },
  "top_risk_customers": [
    { "row_index": 42, "prediction": 1, "churn_probability": 0.97, "label": "Churned", "confidence": 0.97 },
    ...
  ],
  "predictions": [
    { "row_index": 1, "prediction": 0, "churn_probability": 0.12, "label": "Not Churned", "confidence": 0.88 },
    ...
  ]
}
```

**Aggregate Fields:**
| Field | Description |
|-------|-------------|
| `probability_distribution` | 10-bin histogram of churn probabilities |
| `confidence_breakdown` | Count of customers in Low (≤40%), Medium (40-70%), High (>70%) risk |
| `top_risk_customers` | Top 10 highest-risk customers sorted by probability |

---

## `POST /api/v1/explain`

Get SHAP-based feature importance explanation for a single prediction.

**Query Parameters:** Same as `/predict`.

**Request Body:** Same as `/predict` (single customer features).

**Response:**
```json
{
  "prediction": {
    "prediction": 1,
    "churn_probability": 0.82,
    "label": "Churned",
    "confidence": 0.82,
    "timestamp": "..."
  },
  "feature_importances": [
    { "feature": "tenure", "importance": 0.0834, "shap_value": -0.234 },
    { "feature": "Contract_Month-to-month", "importance": 0.0721, "shap_value": 0.187 },
    ...
  ],
  "top_positive": [
    { "feature": "Contract_Month-to-month", "importance": 0.072, "shap_value": 0.187 }
  ],
  "top_negative": [
    { "feature": "tenure", "importance": 0.083, "shap_value": -0.234 }
  ]
}
```

**SHAP Details:**
- `shap_value > 0` — pushes prediction **toward** churn
- `shap_value < 0` — pushes prediction **away from** churn
- `top_positive` — features most responsible for predicting churn
- `top_negative` — features most protective against churn

---

## `GET /api/v1/model/info`

Returns metadata about the currently loaded model.

**Response:**
```json
{
  "model_name": "churn-model-telco",
  "model_version": "1",
  "stage": "Production",
  "metrics": {
    "accuracy": 0.87,
    "f1_score": 0.82,
    "auc_roc": 0.91,
    "precision": 0.84,
    "recall": 0.80
  },
  "features_count": 140,
  "trained_at": "2026-03-15T14:30:00"
}
```

---

## Error Responses

All errors return a standard format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

| Status | Meaning |
|--------|---------|
| `400` | Bad request (invalid file, empty CSV, wrong format) |
| `422` | Validation error (missing/invalid fields) |
| `500` | Internal server error (model not loaded, processing failure) |

---

## Interactive Docs

FastAPI auto-generates interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
