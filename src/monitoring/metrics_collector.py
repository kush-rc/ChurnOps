"""
Prometheus Metrics Collector
============================
Exposes model and API metrics for Prometheus scraping.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ---- API Metrics ----
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ---- Prediction Metrics ----
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total predictions made",
    ["prediction_label"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time to make a prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

BATCH_SIZE = Histogram(
    "batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 50, 100, 500, 1000],
)

# ---- Model Metrics ----
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy",
)

MODEL_F1 = Gauge(
    "model_f1_score",
    "Current model F1 score",
)

MODEL_AUC = Gauge(
    "model_auc_score",
    "Current model AUC-ROC score",
)

# ---- Drift Metrics ----
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current data drift score (PSI)",
)

DRIFT_DETECTED = Gauge(
    "drift_detected",
    "Whether data drift has been detected (0 or 1)",
)

# ---- Model Info ----
MODEL_INFO = Info(
    "model",
    "Information about the currently loaded model",
)
