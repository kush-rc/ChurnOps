"""Unit tests for model evaluation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import ModelEvaluator


class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_returns_all_metrics(self, sample_features):
        """Verify evaluate returns all expected metrics."""
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X.values, y.values)

        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_metrics_in_valid_range(self, sample_features):
        """Verify metrics are within valid ranges."""
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X.values, y.values)

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0 <= metrics[metric] <= 1, f"{metric} out of range: {metrics[metric]}"

    def test_quality_gate_pass(self):
        """Test quality gate with passing metrics."""
        evaluator = ModelEvaluator()
        metrics = {"roc_auc": 0.90, "f1": 0.75, "log_loss": 0.35}
        thresholds = {"min_roc_auc": 0.80, "min_f1": 0.60, "max_log_loss": 0.50}

        passed, failures = evaluator.check_quality_gate(metrics, thresholds)
        assert passed
        assert len(failures) == 0

    def test_quality_gate_fail(self):
        """Test quality gate with failing metrics."""
        evaluator = ModelEvaluator()
        metrics = {"roc_auc": 0.70, "f1": 0.45, "log_loss": 0.60}
        thresholds = {"min_roc_auc": 0.80, "min_f1": 0.60, "max_log_loss": 0.50}

        passed, failures = evaluator.check_quality_gate(metrics, thresholds)
        assert not passed
        assert len(failures) > 0
