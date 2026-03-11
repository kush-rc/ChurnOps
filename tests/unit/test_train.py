"""Unit tests for model training."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class TestModelTraining:
    """Tests for model training."""

    def test_random_forest_trains(self, sample_features):
        """Verify a Random Forest model can train on sample data."""
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        assert acc > 0.5, f"Model accuracy too low: {acc}"

    def test_predictions_are_binary(self, sample_features):
        """Verify predictions are 0 or 1."""
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_in_range(self, sample_features):
        """Verify probabilities are between 0 and 1."""
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        probas = model.predict_proba(X)[:, 1]
        assert all(0 <= p <= 1 for p in probas)
