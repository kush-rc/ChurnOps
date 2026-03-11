"""
Performance Tracker Module
==========================
Tracks model performance metrics over time using a sliding window.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.utils.config import get_monitoring_config


class PerformanceTracker:
    """Tracks and monitors real-time model performance metrics."""

    def __init__(self):
        config = get_monitoring_config()
        perf_config = config.get("performance", {})

        self.window_size = perf_config.get("window_size", 1000)
        self.min_samples = perf_config.get("min_samples", 100)
        self.auc_threshold = perf_config.get("auc_drop_threshold", 0.05)

        self.predictions: deque = deque(maxlen=self.window_size)
        self.actuals: deque = deque(maxlen=self.window_size)
        self.timestamps: deque = deque(maxlen=self.window_size)
        self.baseline_metrics: dict[str, float] = {}

    def set_baseline(self, metrics: dict[str, float]) -> None:
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set: {metrics}")

    def log_prediction(
        self, prediction: float, actual: float | None = None
    ) -> None:
        """Log a prediction (and optionally the actual outcome)."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now().isoformat())

    def get_current_metrics(self) -> dict[str, Any]:
        """Calculate current performance metrics from the sliding window."""
        actuals_available = [a for a in self.actuals if a is not None]

        if len(actuals_available) < self.min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(actuals_available),
                "required": self.min_samples,
            }

        preds = list(self.predictions)[-len(actuals_available):]
        binary_preds = [1 if p >= 0.5 else 0 for p in preds]

        tp = sum(1 for p, a in zip(binary_preds, actuals_available) if p == 1 and a == 1)
        tn = sum(1 for p, a in zip(binary_preds, actuals_available) if p == 0 and a == 0)
        fp = sum(1 for p, a in zip(binary_preds, actuals_available) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(binary_preds, actuals_available) if p == 0 and a == 1)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "status": "ok",
            "samples": len(actuals_available),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_predictions": len(self.predictions),
        }

    def check_performance_degradation(self) -> dict[str, Any]:
        """Check if model performance has degraded below thresholds."""
        current = self.get_current_metrics()
        if current.get("status") != "ok":
            return {"degraded": False, "reason": "insufficient_data"}

        alerts = []
        for metric, threshold_key in [
            ("accuracy", "accuracy_drop_threshold"),
            ("f1", "f1_drop_threshold"),
        ]:
            baseline_val = self.baseline_metrics.get(metric, 0)
            current_val = current.get(metric, 0)
            drop = baseline_val - current_val

            if drop > self.auc_threshold:
                alerts.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "drop": drop,
                })

        return {
            "degraded": len(alerts) > 0,
            "alerts": alerts,
            "current_metrics": current,
        }
