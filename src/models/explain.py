"""
Model Explainability Module
============================
Uses SHAP to explain model predictions.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger

from src.utils.helpers import timer


class ModelExplainer:
    """Generates SHAP-based explanations for model predictions."""

    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray):
        """Initialize the explainer.

        Args:
            model: Trained model (sklearn-compatible).
            X_train: Training data for background distribution.
        """
        self.model = model
        self.explainer = None
        self._create_explainer(X_train)

    def _create_explainer(self, X_train: pd.DataFrame | np.ndarray) -> None:
        """Create the appropriate SHAP explainer."""
        model_type = type(self.model).__name__

        try:
            if model_type in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]:
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type in ["RandomForestClassifier"]:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for any other model
                background = shap.sample(X_train, min(100, len(X_train)))
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )

            logger.info(f"Created SHAP explainer for {model_type}")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            self.explainer = None

    @timer
    def explain(
        self, X: pd.DataFrame | np.ndarray, max_samples: int = 100
    ) -> shap.Explanation | None:
        """Generate SHAP explanations.

        Args:
            X: Feature matrix to explain.
            max_samples: Maximum number of samples to explain.

        Returns:
            SHAP Explanation object.
        """
        if self.explainer is None:
            logger.error("No SHAP explainer available")
            return None

        if len(X) > max_samples:
            X = X[:max_samples] if isinstance(X, np.ndarray) else X.iloc[:max_samples]

        shap_values = self.explainer.shap_values(X)
        logger.info(f"Generated SHAP values for {len(X)} samples")

        return shap_values

    def explain_single(
        self, features: pd.DataFrame | np.ndarray
    ) -> dict:
        """Explain a single prediction.

        Args:
            features: Single row of features.

        Returns:
            Dictionary with feature importances and SHAP values.
        """
        if self.explainer is None:
            return {"error": "No explainer available"}

        if isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
            features_array = features.values
        else:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            features_array = features

        shap_values = self.explainer.shap_values(features_array)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-output: take the positive class
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            values = shap_values[0]

        # Create feature importance ranking
        importance = sorted(
            zip(feature_names, values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "feature_importances": [
                {"feature": name, "shap_value": float(val)}
                for name, val in importance[:20]
            ],
            "top_positive": [
                {"feature": name, "shap_value": float(val)}
                for name, val in importance if val > 0
            ][:5],
            "top_negative": [
                {"feature": name, "shap_value": float(val)}
                for name, val in importance if val < 0
            ][:5],
        }

    def get_feature_importance(
        self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Get global feature importance from SHAP values.

        Returns:
            DataFrame with features ranked by mean |SHAP value|.
        """
        shap_values = self.explain(X)
        if shap_values is None:
            return pd.DataFrame()

        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values = shap_values

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return importance_df
