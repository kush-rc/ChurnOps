"""Unit tests for prediction module."""

from src.models.demo_model import DemoPredictor


class TestPrediction:
    """Tests for prediction module via DemoPredictor."""

    def test_telco_prediction_output_format(self):
        """Verify telco prediction returns complete result dict."""
        features = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "MonthlyCharges": 70.35,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "PaperlessBilling": "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
        }
        result = DemoPredictor.predict("telco", features)

        assert result["prediction"] in (0, 1)
        assert 0 <= result["churn_probability"] <= 1
        assert result["label"] in ("Churned", "Not Churned")

    def test_high_risk_customer_has_elevated_probability(self):
        """Month-to-month + short tenure + fiber optic should trend toward churn."""
        high_risk = {
            "SeniorCitizen": 1,
            "tenure": 1,
            "MonthlyCharges": 110,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
        }
        low_risk = {
            "SeniorCitizen": 0,
            "tenure": 60,
            "MonthlyCharges": 30,
            "Contract": "Two year",
            "InternetService": "DSL",
        }

        result_high = DemoPredictor.predict("telco", high_risk)
        result_low = DemoPredictor.predict("telco", low_risk)

        # High risk should generally have higher churn probability
        assert result_high["churn_probability"] > result_low["churn_probability"]

    def test_prediction_handles_missing_features(self):
        """Predict works even with partial features (others default)."""
        result = DemoPredictor.predict("telco", {"tenure": 12})
        assert "churn_probability" in result

    def test_banking_prediction(self):
        """Verify banking domain predictions work."""
        features = {
            "CreditScore": 650,
            "AccountBalance": 15000,
            "LoanStatus": "Yes",
            "TransactionFrequency": 15,
            "ProductCount": 2,
        }
        result = DemoPredictor.predict("banking", features)
        assert 0 <= result["churn_probability"] <= 1
