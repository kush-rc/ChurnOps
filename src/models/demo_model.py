"""
Demo Model Module
=================
Provides lightweight, pre-trained models for demo/deployment scenarios
where the full MLflow pipeline and GPU-trained models are unavailable.

Each domain gets a small RandomForest trained on realistic synthetic data
so that predictions reflect actual learned patterns rather than random numbers.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Domain feature schemas -- mirrors what each Streamlit form collects
# ---------------------------------------------------------------------------

DOMAIN_SCHEMAS: dict[str, dict[str, Any]] = {
    "telco": {
        "numerical": ["SeniorCitizen", "tenure", "MonthlyCharges"],
        "categorical": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
    },
    "banking": {
        "numerical": ["CreditScore", "AccountBalance", "TransactionFrequency", "ProductCount"],
        "categorical": ["LoanStatus"],
    },
    "ecommerce": {
        "numerical": ["AppActivityScore", "Returns", "OrderFrequency", "DaysSinceLastPurchase"],
        "categorical": ["DiscountUsage"],
    },
    "gaming": {
        "numerical": [
            "WinRate",
            "DailyActiveMinutes",
            "SocialConnections",
            "LevelProgress",
            "InAppPurchasesUSD",
        ],
        "categorical": [],
    },
    "ott": {
        "numerical": ["WatchHours", "LoginFrequency", "PaymentFailures"],
        "categorical": ["GenrePreference", "PlanType"],
    },
    "healthcare": {
        "numerical": ["Age", "HealthScore", "TenureYears", "ClaimsHistory"],
        "categorical": ["PremiumRegularity"],
    },
    "saas": {
        "numerical": ["LoginFrequency", "FeaturesUsed", "SupportTickets", "TeamSize"],
        "categorical": ["BillingCycle"],
    },
    "hospitality": {
        "numerical": ["BookingFrequency", "LoyaltyPoints", "AverageRating", "Complaints"],
        "categorical": ["CityType"],
    },
}

# Mapping from Streamlit form field names → internal schema field names
FORM_FIELD_MAP: dict[str, dict[str, str]] = {
    "telco": {
        "gender": "gender",
        "senior": "SeniorCitizen",
        "partner": "Partner",
        "dependents": "Dependents",
        "tenure": "tenure",
        "monthly_charges": "MonthlyCharges",
        "contract": "Contract",
        "payment": "PaymentMethod",
        "paperless": "PaperlessBilling",
        "phone_service": "PhoneService",
        "multiple_lines": "MultipleLines",
        "internet_service": "InternetService",
        "online_security": "OnlineSecurity",
        "online_backup": "OnlineBackup",
        "device_protection": "DeviceProtection",
        "tech_support": "TechSupport",
        "streaming_tv": "StreamingTV",
        "streaming_movies": "StreamingMovies",
    },
    "banking": {
        "credit_score": "CreditScore",
        "balance": "AccountBalance",
        "loan_status": "LoanStatus",
        "txn_freq": "TransactionFrequency",
        "products": "ProductCount",
    },
    "ecommerce": {
        "app_activity": "AppActivityScore",
        "returns": "Returns",
        "discount_usage": "DiscountUsage",
        "order_freq": "OrderFrequency",
        "days_inactive": "DaysSinceLastPurchase",
    },
    "gaming": {
        "win_rate": "WinRate",
        "daily_hours": "DailyActiveMinutes",
        "social_connections": "SocialConnections",
        "level": "LevelProgress",
        "in_app_purchases": "InAppPurchasesUSD",
    },
    "ott": {
        "label_genre": "GenrePreference",
        "watch_time": "WatchHours",
        "ad_tier": "PlanType",
        "login_freq": "LoginFrequency",
        "failed_payments": "PaymentFailures",
    },
    "healthcare": {
        "age": "Age",
        "health_score": "HealthScore",
        "tenure": "TenureYears",
        "claims": "ClaimsHistory",
        "premium_reg": "PremiumRegularity",
    },
    "saas": {
        "login_freq": "LoginFrequency",
        "features_used": "FeaturesUsed",
        "billing": "BillingCycle",
        "team_size": "TeamSize",
        "tickets": "SupportTickets",
    },
    "hospitality": {
        "booking_freq": "BookingFrequency",
        "loyalty": "LoyaltyPoints",
        "city_type": "CityType",
        "ratings": "AverageRating",
        "complaints": "Complaints",
    },
}


# ---------------------------------------------------------------------------
# Synthetic data generators per domain
# ---------------------------------------------------------------------------


def _generate_synthetic(
    domain: str, n: int = 1000, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small realistic synthetic dataset for a domain."""
    rng = np.random.RandomState(seed)
    DOMAIN_SCHEMAS[domain]
    data: dict[str, Any] = {}

    if domain == "telco":
        data["SeniorCitizen"] = rng.choice([0, 1], n, p=[0.8, 0.2])
        data["tenure"] = rng.randint(0, 72, n)
        data["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
        data["gender"] = rng.choice(["Male", "Female"], n)
        data["Partner"] = rng.choice(["Yes", "No"], n)
        data["Dependents"] = rng.choice(["Yes", "No"], n)
        data["PhoneService"] = rng.choice(["Yes", "No"], n)
        data["MultipleLines"] = rng.choice(["Yes", "No", "No phone service"], n)
        data["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], n)
        data["OnlineSecurity"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["OnlineBackup"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["DeviceProtection"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["TechSupport"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["StreamingTV"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["StreamingMovies"] = rng.choice(["Yes", "No", "No internet service"], n)
        data["Contract"] = rng.choice(
            ["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.25, 0.25]
        )
        data["PaperlessBilling"] = rng.choice(["Yes", "No"], n)
        data["PaymentMethod"] = rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n,
        )
        # Realistic churn driver: short tenure + month-to-month + high charges = more churn
        churn_score = (
            (72 - data["tenure"]) / 72 * 0.4
            + (np.array(data["Contract"]) == "Month-to-month").astype(float) * 0.3
            + data["MonthlyCharges"] / 120 * 0.2
            + (np.array(data["InternetService"]) == "Fiber optic").astype(float) * 0.1
        )

    elif domain == "banking":
        data["CreditScore"] = rng.randint(300, 850, n)
        data["AccountBalance"] = rng.uniform(0, 500000, n).round(2)
        data["LoanStatus"] = rng.choice(["Yes", "No"], n)
        data["TransactionFrequency"] = rng.randint(0, 100, n)
        data["ProductCount"] = rng.randint(1, 5, n)
        churn_score = (
            (850 - data["CreditScore"]) / 550 * 0.3
            + (data["TransactionFrequency"] < 10).astype(float) * 0.3
            + (data["ProductCount"] <= 1).astype(float) * 0.2
            + (np.array(data["LoanStatus"]) == "Yes").astype(float) * 0.2
        )

    elif domain == "ecommerce":
        data["AppActivityScore"] = rng.randint(0, 100, n)
        data["Returns"] = rng.uniform(0, 100, n).round(1)
        data["DiscountUsage"] = rng.choice(["High", "Medium", "Low"], n)
        data["OrderFrequency"] = rng.randint(0, 30, n)
        data["DaysSinceLastPurchase"] = rng.randint(0, 365, n)
        churn_score = (
            (100 - data["AppActivityScore"]) / 100 * 0.2
            + data["Returns"] / 100 * 0.2
            + data["DaysSinceLastPurchase"] / 365 * 0.3
            + (data["OrderFrequency"] < 3).astype(float) * 0.2
            + (np.array(data["DiscountUsage"]) == "High").astype(float) * 0.1
        )

    elif domain == "gaming":
        data["WinRate"] = rng.uniform(0, 100, n).round(1)
        data["DailyActiveMinutes"] = rng.uniform(0, 1440, n).round(1)
        data["SocialConnections"] = rng.randint(0, 500, n)
        data["LevelProgress"] = rng.randint(1, 100, n)
        data["InAppPurchasesUSD"] = rng.uniform(0, 1000, n).round(2)
        churn_score = (
            (100 - data["WinRate"]) / 100 * 0.2
            + (1440 - data["DailyActiveMinutes"]) / 1440 * 0.3
            + (500 - data["SocialConnections"]) / 500 * 0.2
            + (100 - data["LevelProgress"]) / 100 * 0.2
            + (data["InAppPurchasesUSD"] < 5).astype(float) * 0.1
        )

    elif domain == "ott":
        data["WatchHours"] = rng.uniform(1, 500, n).round(1)
        data["LoginFrequency"] = rng.randint(0, 50, n)
        data["PaymentFailures"] = rng.choice(
            [0, 1, 2, 3, 4, 5], n, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]
        )
        data["GenrePreference"] = rng.choice(["Action", "Comedy", "Drama", "Sci-Fi", "Kids"], n)
        data["PlanType"] = rng.choice(["Mobile", "Basic", "Standard", "Premium"], n)
        churn_score = (
            (500 - data["WatchHours"]) / 500 * 0.3
            + (50 - data["LoginFrequency"]) / 50 * 0.2
            + data["PaymentFailures"] / 5 * 0.3
            + (np.array(data["PlanType"]) == "Mobile").astype(float) * 0.2
        )

    elif domain == "healthcare":
        data["Age"] = rng.randint(18, 90, n)
        data["HealthScore"] = rng.randint(40, 100, n)
        data["TenureYears"] = rng.randint(1, 20, n)
        data["ClaimsHistory"] = rng.randint(0, 20, n)
        data["PremiumRegularity"] = rng.choice(
            ["Regular", "Irregular", "Delayed"], n, p=[0.6, 0.25, 0.15]
        )
        churn_score = (
            (100 - data["HealthScore"]) / 60 * 0.2
            + (20 - data["TenureYears"]) / 19 * 0.2
            + data["ClaimsHistory"] / 20 * 0.2
            + (np.array(data["PremiumRegularity"]) != "Regular").astype(float) * 0.3
            + (data["Age"] < 30).astype(float) * 0.1
        )

    elif domain == "saas":
        data["LoginFrequency"] = rng.randint(0, 100, n)
        data["FeaturesUsed"] = rng.randint(1, 20, n)
        data["BillingCycle"] = rng.choice(["Annual", "Monthly"], n)
        data["TeamSize"] = rng.randint(1, 100, n)
        data["SupportTickets"] = rng.randint(0, 20, n)
        churn_score = (
            (100 - data["LoginFrequency"]) / 100 * 0.3
            + (20 - data["FeaturesUsed"]) / 19 * 0.2
            + (np.array(data["BillingCycle"]) == "Monthly").astype(float) * 0.2
            + data["SupportTickets"] / 20 * 0.2
            + (data["TeamSize"] <= 2).astype(float) * 0.1
        )

    elif domain == "hospitality":
        data["BookingFrequency"] = rng.randint(0, 50, n)
        data["LoyaltyPoints"] = rng.randint(0, 10000, n)
        data["CityType"] = rng.choice(["Tier 1", "Tier 2", "Tier 3", "International"], n)
        data["AverageRating"] = rng.uniform(1, 5, n).round(1)
        data["Complaints"] = rng.choice([0, 1, 2, 3], n, p=[0.5, 0.3, 0.15, 0.05])
        churn_score = (
            (50 - data["BookingFrequency"]) / 50 * 0.2
            + (10000 - data["LoyaltyPoints"]) / 10000 * 0.2
            + (5 - data["AverageRating"]) / 4 * 0.2
            + data["Complaints"] / 3 * 0.3
            + (np.array(data["CityType"]) == "Tier 3").astype(float) * 0.1
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Convert score to binary churn with some noise
    noise = rng.normal(0, 0.12, n)
    churn_prob = np.clip(churn_score + noise, 0, 1)
    y = (churn_prob > 0.55).astype(int)

    df = pd.DataFrame(data)
    return df, pd.Series(y, name="Churn")


# ---------------------------------------------------------------------------
# Demo Predictor
# ---------------------------------------------------------------------------


class DemoPredictor:
    """Lightweight predictor using small RandomForest models trained on synthetic data.

    Models are trained once on first use and cached in memory.
    This replaces the random-number spoof with actual learned patterns.
    """

    _models: dict[str, RandomForestClassifier] = {}
    _encoders: dict[str, dict[str, LabelEncoder]] = {}

    @classmethod
    def _ensure_model(cls, domain: str) -> None:
        """Train and cache a model for the given domain if not already present."""
        if domain in cls._models:
            return

        schema = DOMAIN_SCHEMAS.get(domain)
        if schema is None:
            raise ValueError(f"No schema for domain: {domain}")

        df, y = _generate_synthetic(domain, n=1000, seed=42)

        # Encode categoricals
        encoders: dict[str, LabelEncoder] = {}
        for col in schema["categorical"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        # Ensure column order matches predict()
        ordered_cols = schema["numerical"] + schema["categorical"]
        df = df[ordered_cols]

        model = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(df, y)

        cls._models[domain] = model
        cls._encoders[domain] = encoders

    @classmethod
    def predict(cls, domain: str, features: dict[str, Any]) -> dict[str, Any]:
        """Predict churn for a single customer.

        Args:
            domain: Industry domain key (e.g. 'telco', 'banking').
            features: Dict of feature name -> value.
                      Names should match DOMAIN_SCHEMAS keys.

        Returns:
            Dict with prediction, churn_probability, label, confidence.
        """
        cls._ensure_model(domain)
        schema = DOMAIN_SCHEMAS[domain]
        model = cls._models[domain]
        encoders = cls._encoders[domain]

        # Build a single-row DataFrame in the correct column order
        row: dict[str, Any] = {}
        for col in schema["numerical"]:
            val = features.get(col, 0)
            row[col] = float(val) if val is not None else 0.0

        for col in schema["categorical"]:
            val = features.get(col, "Unknown")
            if val is None:
                val = "Unknown"
            le = encoders[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                # Unseen category → use first class as fallback
                row[col] = 0

        df = pd.DataFrame([row])
        proba = model.predict_proba(df)[0, 1]
        prediction = int(proba >= 0.5)

        return {
            "prediction": prediction,
            "churn_probability": round(float(proba), 4),
            "label": "Churned" if prediction == 1 else "Not Churned",
            "confidence": round(float(max(proba, 1 - proba)), 4),
        }

    @classmethod
    def get_feature_importances(cls, domain: str) -> list[dict[str, Any]]:
        """Return feature importances for a domain model."""
        cls._ensure_model(domain)
        schema = DOMAIN_SCHEMAS[domain]
        model = cls._models[domain]
        feature_names = schema["numerical"] + schema["categorical"]
        importances = model.feature_importances_

        results = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(feature_names, importances, strict=False)
        ]
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    @classmethod
    def map_form_fields(cls, domain: str, form_data: dict[str, Any]) -> dict[str, Any]:
        """Map Streamlit form field names to internal schema field names.

        Args:
            domain: Industry domain key.
            form_data: Dict with Streamlit form variable names as keys.

        Returns:
            Dict with schema-compatible field names.
        """
        field_map = FORM_FIELD_MAP.get(domain, {})
        mapped: dict[str, Any] = {}
        for form_key, schema_key in field_map.items():
            if form_key in form_data:
                mapped[schema_key] = form_data[form_key]
        # Also pass through any keys already matching schema names
        schema = DOMAIN_SCHEMAS.get(domain, {})
        all_fields = schema.get("numerical", []) + schema.get("categorical", [])
        for key in all_fields:
            if key in form_data and key not in mapped:
                mapped[key] = form_data[key]
        return mapped
