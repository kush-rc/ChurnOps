"""
Model Training Module
=====================
Trains multiple ML models, logs to MLflow, and handles
class imbalance with SMOTE/ADASYN.
"""

from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.models.evaluate import ModelEvaluator
from src.utils.config import get_config, get_model_config, get_training_config
from src.utils.helpers import timer


class ModelTrainer:
    """Trains multiple models and logs experiments to MLflow."""

    def __init__(self):
        self.training_config = get_training_config()
        self.main_config = get_config()
        self.evaluator = ModelEvaluator()
        self.models: dict[str, Any] = {}
        self.results: list[dict] = []

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        mlflow_config = self.main_config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
        experiment_name = mlflow_config.get("experiment_name", "churn-prediction")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")

    def _get_model_instance(self, model_name: str) -> Any:
        """Create a model instance from config."""
        config = get_model_config(model_name)
        params = config.get("params", {})

        model_map = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "neural_net": MLPClassifier,
        }

        if model_name in model_map:
            return model_map[model_name](**params)

        # Import gradient boosting models dynamically
        if model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**params)

        raise ValueError(f"Unknown model: {model_name}")

    def _handle_imbalance(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply class imbalance handling."""
        strategy = self.training_config.get("imbalance_strategy", "smote")
        sampling = self.training_config.get("smote_sampling_strategy", 0.8)
        seed = self.training_config.get("random_state", 42)

        if strategy == "none":
            return X, y

        sampler_map = {
            "smote": SMOTE(sampling_strategy=sampling, random_state=seed),
            "adasyn": ADASYN(sampling_strategy=sampling, random_state=seed),
            "borderline_smote": BorderlineSMOTE(sampling_strategy=sampling, random_state=seed),
        }

        sampler = sampler_map.get(strategy)
        if sampler is None:
            logger.warning(f"Unknown imbalance strategy: {strategy}")
            return X, y

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(
            f"Applied {strategy}: {len(X)} → {len(X_resampled)} samples"
        )
        return X_resampled, y_resampled

    @timer
    def train_all(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Train all configured models.

        Args:
            X: Feature matrix.
            y: Target variable.

        Returns:
            Dictionary with model names and their evaluation results.
        """
        self._setup_mlflow()

        seed = self.training_config.get("random_state", 42)
        test_size = self.training_config.get("test_size", 0.2)
        models_to_train = self.training_config.get("models", [])
        primary_metric = self.training_config.get("primary_metric", "roc_auc")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Handle class imbalance on training data only
        X_train_balanced, y_train_balanced = self._handle_imbalance(
            X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            y_train.values if isinstance(y_train, pd.Series) else y_train,
        )

        # Train each model
        self.results = []
        for model_name in models_to_train:
            try:
                result = self._train_single_model(
                    model_name, X_train_balanced, y_train_balanced,
                    X_test, y_test, primary_metric
                )
                self.results.append(result)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        # Sort by primary metric
        self.results.sort(key=lambda r: r.get(primary_metric, 0), reverse=True)

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL COMPARISON RESULTS")
        logger.info("=" * 60)
        for r in self.results:
            logger.info(
                f"  {r['model_name']:25s} | "
                f"AUC: {r.get('roc_auc', 0):.4f} | "
                f"F1: {r.get('f1', 0):.4f} | "
                f"Accuracy: {r.get('accuracy', 0):.4f}"
            )
        logger.info("=" * 60)

        best = self.results[0] if self.results else None
        if best:
            logger.info(f"🏆 Best model: {best['model_name']} (AUC: {best.get('roc_auc', 0):.4f})")

        return {"results": self.results, "best_model": best}

    def _train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Any,
        y_test: Any,
        primary_metric: str,
    ) -> dict:
        """Train a single model and log to MLflow."""
        logger.info(f"\n🚀 Training: {model_name}")

        model = self._get_model_instance(model_name)
        config = get_model_config(model_name)

        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_params(config.get("params", {}))
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("imbalance_strategy", self.training_config.get("imbalance_strategy"))

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

            metrics = self.evaluator.evaluate(model, X_test_np, y_test_np)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Store model
            self.models[model_name] = model

            result = {"model_name": model_name, "run_id": run.info.run_id, **metrics}
            logger.info(f"  AUC: {metrics.get('roc_auc', 0):.4f} | F1: {metrics.get('f1', 0):.4f}")

            return result


if __name__ == "__main__":
    from src.utils.config import get_dataset_config, get_path
    from src.utils.helpers import load_dataframe

    config = get_dataset_config()
    features_path = get_path("data_features") / f"{config['name']}_features.parquet"
    df = load_dataframe(features_path)

    target = config["target"]
    X = df.drop(columns=[target])
    y = df[target]

    trainer = ModelTrainer()
    results = trainer.train_all(X, y)
