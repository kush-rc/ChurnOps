"""
Prediction Module
=================
Loads a trained model and makes predictions on new data.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.data.features import FeatureEngineer
from src.data.preprocess import DataPreprocessor
from src.utils.config import get_config
from src.utils.helpers import timer


class ChurnPredictor:
    """Makes churn predictions using a trained model from MLflow."""

    def __init__(self, domain: str, model_uri: str | None = None):
        """Initialize the predictor.

        Args:
            domain: Industry domain (e.g., 'telco', 'banking').
            model_uri: MLflow model URI (e.g., 'models:/telco-churn-model/Production').
        """
        self.domain = domain
        self.model = None
        self.model_uri = model_uri

        # Inference pipeline components
        self.preprocessor: DataPreprocessor | None = None
        self.engineer: FeatureEngineer | None = None

        # SHAP caching (built lazily on first explain request)
        self._shap_explainer = None
        self._shap_baseline = None

        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load model from MLflow and preprocessing states from disk."""
        import warnings
        # Suppress MLflow dependency mismatch warnings for cleaner/faster logs
        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

        config = get_config()

        # Try to load bundled deployment model first (fastest, no DB needed)
        bundled_path = Path(f"models/production/{self.domain}")
        if bundled_path.exists():
            try:
                self.model_uri = str(bundled_path.absolute())
                self.model = mlflow.pyfunc.load_model(bundled_path.as_posix())
                logger.info(f"Loaded bundled model from: {bundled_path}")
            except Exception as e:
                logger.warning(f"Failed to load bundled model: {e}")

        # Fallback to MLflow tracking server only if no bundled model
        if not self.model:
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            client = mlflow.tracking.MlflowClient()
            model_name = f"{self.domain}-churn-model"
            try:
                latest = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
                if latest:
                    latest_version = sorted(latest, key=lambda v: int(v.version))[-1]
                    self.model_uri = f"models:/{model_name}/{latest_version.version}"
                    self.model = mlflow.pyfunc.load_model(self.model_uri)
                    logger.info(f"Loaded latest model: {self.model_uri}")
                else:
                    logger.warning(f"No registered model found for {model_name}")
            except Exception as e:
                logger.warning(f"Could not load MLflow model for {model_name}: {e}")
        # Load Preprocessor & Engineer States
        try:
            self.preprocessor = DataPreprocessor.load_state(self.domain)
            self.engineer = FeatureEngineer.load_state(self.domain)
            logger.info(f"Loaded inference pipeline state for domain: {self.domain}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load pipeline state: {e}. Run orchestrate_all.ps1 first.")

    def load_model(self, model_uri: str) -> None:
        """Load a specific model from MLflow."""
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
        """Get churn probability scores."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # 1. Try to find the underlying model with predict_proba
        underlying = None
        if hasattr(self.model, '_model_impl'):
            impl = self.model._model_impl
            # Check common MLflow wrapper attributes
            for attr in ['sklearn_model', 'xgb_model', 'lgbm_model']:
                if hasattr(impl, attr):
                    underlying = getattr(impl, attr)
                    break
            if not underlying:
                underlying = impl

        if underlying and hasattr(underlying, 'predict_proba'):
            try:
                # Try with whatever input we have (DF or Array)
                probas = underlying.predict_proba(X)
                return probas[:, 1] if probas.ndim > 1 else probas
            except Exception as e:
                # If it failed due to names, try stripping them if it's a DataFrame
                if isinstance(X, pd.DataFrame):
                    logger.warning(f"Predict proba failed on DataFrame, trying numpy: {e}")
                    probas = underlying.predict_proba(X.to_numpy())
                    return probas[:, 1] if probas.ndim > 1 else probas
                raise

        # 2. Fallback to binary predict if no proba
        try:
            preds = self.model.predict(X)
            return preds.astype(float)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _extract_model_feature_names(self) -> list[str] | None:
        """Intelligently extract expected feature names from the model wrapper."""
        if not self.model or not hasattr(self.model, '_model_impl'):
            return None

        impl = self.model._model_impl

        # 1. Check for standard sklearn feature_names_in_
        if hasattr(impl, 'feature_names_in_'):
            return list(impl.feature_names_in_)

        # 2. Check underlying sklearn_model or xgb_model
        for attr in ['sklearn_model', 'xgb_model']:
            if hasattr(impl, attr):
                sub = getattr(impl, attr)
                if hasattr(sub, 'feature_names_in_'):
                    return list(sub.feature_names_in_)
                if hasattr(sub, 'get_booster'):
                    return sub.get_booster().feature_names
                if hasattr(sub, 'feature_names'):
                    return sub.feature_names

        # 3. Fallback to engineer's training names
        return self.engineer.feature_names if self.engineer else None

    def process_and_predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Run data through the full inference pipeline."""
        if self.preprocessor is None or self.engineer is None:
            raise RuntimeError("Pipeline state not loaded. Cannot process raw data.")

        # 1. Preprocess (Imputes missing columns and cleans types)
        df_clean = self.preprocessor.preprocess(df)

        # 2. Engineer features (Derived features, interactions, OHE)
        df_features = self.engineer.engineer_features(df_clean)

        # Ensure column order matches training exactly and fill missing with 0
        expected_cols = self._extract_model_feature_names() or self.engineer.feature_names

        # Reindex automatically adds missing columns (filled with 0) and drops extra ones,
        # while strictly enforcing the exact column order expected by the ML model.
        X = df_features.reindex(columns=expected_cols, fill_value=0)

        # 3. Predict with robust fallback
        try:
            # First try with DataFrame to maintain feature names if supported
            proba = self.predict_proba(X)
        except Exception as e:
            logger.warning(f"DataFrame prediction failed ({e}). Falling back to numpy array.")
            # Fallback to raw numpy array to bypass strict column name validation
            # while keeping the data in the exactly correct order.
            proba = self.predict_proba(X.to_numpy())

        pred = (proba >= 0.5).astype(int)

        return pred, proba

    def predict_single(self, features: dict) -> dict:
        """Predict churn for a single customer."""
        df = pd.DataFrame([features])

        pred, proba = self.process_and_predict(df)

        prediction = int(pred[0])
        float(proba[0])

        return {
            "prediction": prediction,
            "churn_probability": float(proba),
            "label": "Churned" if prediction == 1 else "Not Churned",
            "confidence": float(max(proba, 1 - proba)),
        }

    def explain_single(self, features: dict) -> list[dict]:
        """Compute exact SHAP values for a single prediction using the real ML model."""
        if not self.model or not self.preprocessor or not self.engineer:
            return [{"feature": "Unknown", "shap_value": 0.0}]

        import shap

        df = pd.DataFrame([features])
        df_clean = self.preprocessor.preprocess(df)
        df_features = self.engineer.engineer_features(df_clean)

        expected_cols = self._extract_model_feature_names() or self.engineer.feature_names
        X = df_features.reindex(columns=expected_cols, fill_value=0)

        # Extract the underlying XGBoost/sklearn model from the MLflow wrapper
        impl = self.model._model_impl
        underlying = None
        for attr in ['xgb_model', 'sklearn_model', 'lgbm_model']:
            if hasattr(impl, attr):
                underlying = getattr(impl, attr)
                break
        if not underlying:
            underlying = impl

        try:
            # ─── Strategy 1: Patch the XGBoost Booster directly ───
            # XGBoost >= 2.0 stores base_score as '[6.9498456E-1]' in the booster
            # config JSON, which shap's C-extension parser can't convert to float.
            # Fix: extract the booster, save to temp JSON, fix the string, reload.
            if hasattr(underlying, 'get_booster'):
                import json
                import os
                import tempfile

                booster = underlying.get_booster()

                # Save the booster to a temp JSON file
                tmp_path = os.path.join(tempfile.gettempdir(), '_shap_xgb_fix.json')
                booster.save_model(tmp_path)

                with open(tmp_path) as f:
                    model_json = json.load(f)

                # Fix the base_score string: '[6.9498456E-1]' -> '6.9498456E-1'
                try:
                    bs = model_json['learner']['learner_model_param']['base_score']
                    if isinstance(bs, str) and ('[' in bs or ']' in bs):
                        model_json['learner']['learner_model_param']['base_score'] = bs.replace('[', '').replace(']', '')

                        with open(tmp_path, 'w') as f:
                            json.dump(model_json, f)

                        booster.load_model(tmp_path)
                        logger.info("Patched XGBoost base_score for SHAP compatibility")
                except (KeyError, TypeError):
                    pass  # Not an XGBoost JSON format we expected

                # Use the patched booster directly with TreeExplainer
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(X)

                # Handle multi-class output (list of arrays)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    sv = shap_values[1][0]
                elif isinstance(shap_values, np.ndarray):
                    sv = shap_values[0] if shap_values.ndim == 2 else shap_values
                else:
                    sv = shap_values[0]

                if hasattr(sv, 'ndim') and sv.ndim > 1:
                    sv = sv[:, 1]

                importances = [
                    {"feature": str(col), "shap_value": round(float(val), 4)}
                    for col, val in zip(X.columns, sv, strict=False)
                ]
            else:
                # Non-XGBoost model: try TreeExplainer directly
                explainer = shap.TreeExplainer(underlying)
                shap_values = explainer.shap_values(X)

                if isinstance(shap_values, list) and len(shap_values) == 2:
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]
                    if hasattr(sv, 'ndim') and sv.ndim > 1:
                        sv = sv[:, 1]

                importances = [
                    {"feature": str(col), "shap_value": round(float(val), 4)}
                    for col, val in zip(X.columns, sv, strict=False)
                ]

        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to PermutationExplainer: {e}")
            try:
                # Fallback: use predict_proba as a callable with a zero baseline
                # Cache the explainer for subsequent requests (first call ~10s, subsequent ~0.5s)
                if self._shap_explainer is None:
                    predict_fn = underlying.predict_proba if hasattr(underlying, 'predict_proba') else underlying.predict
                    self._shap_baseline = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)
                    self._shap_explainer = shap.Explainer(predict_fn, self._shap_baseline)
                    logger.info("Built and cached SHAP PermutationExplainer")

                shap_values_obj = self._shap_explainer(X)
                sv = shap_values_obj.values[0]

                if sv.ndim > 1:
                    sv = sv[:, 1]

                importances = [
                    {"feature": str(col), "shap_value": round(float(val), 4)}
                    for col, val in zip(X.columns, sv, strict=False)
                ]
            except Exception as inner_e:
                logger.error(f"Failed to generate SHAP values: {inner_e}")
                # Ultimate fallback: use global feature importances from the model
                if hasattr(underlying, 'feature_importances_'):
                    importances = [
                        {"feature": str(col), "shap_value": round(float(imp), 4)}
                        for col, imp in zip(X.columns, underlying.feature_importances_, strict=False)
                    ]
                else:
                    return [{"feature": "Unknown", "shap_value": 0.0}]

        # Sort by absolute impact
        importances.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return importances
