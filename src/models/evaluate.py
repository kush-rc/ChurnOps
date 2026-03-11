"""
Model Evaluation Module
=======================
Computes comprehensive evaluation metrics for binary classification models.
Generates confusion matrices, ROC curves, and PR curves.
"""

from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.helpers import timer


class ModelEvaluator:
    """Evaluates binary classification models with comprehensive metrics."""

    @timer
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            model: Trained sklearn-compatible model.
            X_test: Test features.
            y_test: True labels.

        Returns:
            Dictionary of metric name → value.
        """
        y_pred = model.predict(X_test)

        # Get probability predictions if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["average_precision"] = average_precision_score(y_test, y_proba)
            metrics["log_loss"] = log_loss(y_test, y_proba)

        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def get_classification_report(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> str:
        """Get a formatted classification report.

        Returns:
            Formatted classification report string.
        """
        y_pred = model.predict(X_test)
        return classification_report(
            y_test, y_pred, target_names=["Not Churned", "Churned"]
        )

    def check_quality_gate(
        self, metrics: dict[str, float], thresholds: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """Check if model metrics pass quality gates.

        Args:
            metrics: Computed metrics dictionary.
            thresholds: Minimum thresholds per metric.

        Returns:
            Tuple of (passed, list of failure messages).
        """
        failures = []

        gate_checks = {
            "min_roc_auc": "roc_auc",
            "min_f1": "f1",
            "max_log_loss": "log_loss",
        }

        for gate_key, metric_key in gate_checks.items():
            threshold = thresholds.get(gate_key)
            if threshold is None:
                continue

            value = metrics.get(metric_key)
            if value is None:
                continue

            if gate_key.startswith("min_") and value < threshold:
                failures.append(f"{metric_key}={value:.4f} < {threshold}")
            elif gate_key.startswith("max_") and value > threshold:
                failures.append(f"{metric_key}={value:.4f} > {threshold}")

        passed = len(failures) == 0
        if passed:
            logger.info("✅ Quality gate PASSED")
        else:
            logger.warning(f"❌ Quality gate FAILED: {failures}")

        return passed, failures
