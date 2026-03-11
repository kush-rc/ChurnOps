"""Shared test fixtures and configuration."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_telco_data():
    """Create a sample Telco dataset for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "customerID": [f"CUST_{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "MonthlyCharges": np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges": np.round(np.random.uniform(18, 8700, n), 2),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.26, 0.74]),
    })


@pytest.fixture
def sample_features():
    """Create a sample feature matrix for testing."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame(
        np.random.randn(n, 20),
        columns=[f"feature_{i}" for i in range(20)]
    )
    y = pd.Series(np.random.choice([0, 1], n, p=[0.74, 0.26]), name="Churn")

    return X, y


@pytest.fixture
def sample_customer_features():
    """Create a sample customer features dict for API testing."""
    return {
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
        "TotalCharges": 1397.5,
    }
