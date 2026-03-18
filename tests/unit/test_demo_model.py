"""Unit tests for demo model predictions."""

import pytest
from src.models.demo_model import DemoPredictor, DOMAIN_SCHEMAS


class TestDemoPredictor:
    """Tests for the lightweight demo prediction model."""

    DOMAINS = list(DOMAIN_SCHEMAS.keys())

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_predict_returns_valid_structure(self, domain):
        """Verify prediction output has expected keys and types."""
        schema = DOMAIN_SCHEMAS[domain]
        # Build minimal features
        features = {}
        for col in schema["numerical"]:
            features[col] = 50.0
        for col in schema["categorical"]:
            features[col] = "Unknown"

        result = DemoPredictor.predict(domain, features)

        assert "prediction" in result
        assert "churn_probability" in result
        assert "label" in result
        assert "confidence" in result

        assert result["prediction"] in (0, 1)
        assert 0 <= result["churn_probability"] <= 1
        assert result["label"] in ("Churned", "Not Churned")
        assert 0.5 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_predict_probability_changes_with_input(self, domain):
        """Verify that different inputs produce different probabilities."""
        schema = DOMAIN_SCHEMAS[domain]

        # Low-risk profile
        features_low = {}
        for col in schema["numerical"]:
            features_low[col] = 90.0  # High values
        for col in schema["categorical"]:
            features_low[col] = "Unknown"

        # High-risk profile
        features_high = {}
        for col in schema["numerical"]:
            features_high[col] = 1.0  # Low values
        for col in schema["categorical"]:
            features_high[col] = "Unknown"

        result_low = DemoPredictor.predict(domain, features_low)
        result_high = DemoPredictor.predict(domain, features_high)

        # The probabilities should differ (model learned patterns)
        assert result_low["churn_probability"] != result_high["churn_probability"]

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_get_feature_importances(self, domain):
        """Verify feature importances are returned for all features."""
        importances = DemoPredictor.get_feature_importances(domain)
        schema = DOMAIN_SCHEMAS[domain]
        total_features = len(schema["numerical"]) + len(schema["categorical"])

        assert len(importances) == total_features
        for imp in importances:
            assert "feature" in imp
            assert "importance" in imp
            assert 0 <= imp["importance"] <= 1

    def test_unknown_domain_raises_error(self):
        """Verify ValueError for unknown domains."""
        with pytest.raises(ValueError, match="No schema for domain"):
            DemoPredictor.predict("nonexistent_domain", {})

    def test_model_caching(self):
        """Verify models are cached and not retrained on subsequent calls."""
        DemoPredictor.predict("telco", {"tenure": 12, "MonthlyCharges": 70})
        assert "telco" in DemoPredictor._models

        model_id = id(DemoPredictor._models["telco"])
        DemoPredictor.predict("telco", {"tenure": 24, "MonthlyCharges": 50})
        assert id(DemoPredictor._models["telco"]) == model_id

    def test_map_form_fields(self):
        """Verify form field mapping works correctly."""
        form_data = {"tenure": 12, "monthly_charges": 70.0, "contract": "Monthly"}
        mapped = DemoPredictor.map_form_fields("telco", form_data)
        assert mapped.get("tenure") == 12
        assert mapped.get("MonthlyCharges") == 70.0
