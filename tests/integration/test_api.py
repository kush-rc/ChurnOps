"""Integration test for API."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestAPI:
    """Integration tests for the FastAPI application."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_endpoint(self):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_predict_endpoint(self, sample_customer_features):
        """Test single prediction endpoint."""
        response = client.post("/api/v1/predict", json=sample_customer_features)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "churn_probability" in data

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data

    def test_docs_endpoint(self):
        """Test API docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
