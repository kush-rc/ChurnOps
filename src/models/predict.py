"""
Prediction Module
=================
Loads a trained model and makes predictions on new data.
"""

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import get_config
from src.utils.helpers import timer


class ChurnPredictor:
    """Makes churn predictions using a trained model from MLflow."""

    def __init__(self, model_uri: str | None = None):
        """Initialize the predictor.

        Args:
            model_uri: MLflow model URI (e.g., 'models:/churn-model/Production').
                       If None, loads the latest from the default experiment.
        """
        self.model = None
        self.model_uri = model_uri
        self._load_model()

    def _load_model(self) -> None:
        """Load model from MLflow."""
        config = get_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        if self.model_uri:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info(f"Loaded model from: {self.model_uri}")
        else:
            logger.warning("No model URI specified. Call load_model() with a URI.")

    def load_model(self, model_uri: str) -> None:
        """Load a specific model from MLflow.

        Args:
            model_uri: MLflow model URI.
        """
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model: {model_uri}")

    @timer
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make churn predictions.

        Args:
            X: Feature matrix (single or batch).

        Returns:
            Array of predictions (0 = not churned, 1 = churned).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        predictions = self.model.predict(X)
        logger.info(f"Made {len(predictions)} predictions")
        return predictions

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get churn probability scores.

        Args:
            X: Feature matrix.

        Returns:
            Array of churn probabilities.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # MLflow pyfunc models may not have predict_proba
        unwrapped = self.model._model_impl if hasattr(self.model, '_model_impl') else self.model
        if hasattr(unwrapped, 'predict_proba'):
            probas = unwrapped.predict_proba(X)
            return probas[:, 1] if probas.ndim > 1 else probas
        else:
            return self.predict(X).astype(float)

    def predict_single(self, features: dict) -> dict:
        """Predict churn for a single customer.

        Args:
            features: Dictionary of feature name → value.

        Returns:
            Prediction result with probability and label.
        """
        df = pd.DataFrame([features])
        proba = self.predict_proba(df)[0]
        prediction = int(proba >= 0.5)

        return {
            "prediction": prediction,
            "churn_probability": float(proba),
            "label": "Churned" if prediction == 1 else "Not Churned",
            "confidence": float(max(proba, 1 - proba)),
        }
