"""
Model Registry Module
=====================
Manages model versions in MLflow Model Registry.
Handles model promotion from Staging to Production.
"""

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from src.utils.config import get_config
from src.utils.helpers import timer


class ModelRegistry:
    """Manages model lifecycle in MLflow Model Registry."""

    def __init__(self):
        config = get_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        self.client = MlflowClient()
        self.model_name = "churn-prediction-model"

    @timer
    def register_model(self, run_id: str, model_name: str | None = None) -> str:
        """Register a model from an MLflow run.

        Args:
            run_id: MLflow run ID containing the model.
            model_name: Registry name (default: churn-prediction-model).

        Returns:
            Model version string.
        """
        name = model_name or self.model_name
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, name)
        logger.info(f"Registered model '{name}' version {result.version}")
        return result.version

    def transition_model(
        self, version: str, stage: str, model_name: str | None = None
    ) -> None:
        """Transition a model version to a new stage.

        Args:
            version: Model version number.
            stage: Target stage ('Staging', 'Production', 'Archived').
            model_name: Registry name.
        """
        name = model_name or self.model_name
        self.client.transition_model_version_stage(
            name=name, version=version, stage=stage
        )
        logger.info(f"Model '{name}' v{version} → {stage}")

    def get_production_model_uri(self, model_name: str | None = None) -> str | None:
        """Get the URI of the current production model.

        Returns:
            Model URI string, or None if no production model exists.
        """
        name = model_name or self.model_name
        try:
            versions = self.client.get_latest_versions(name, stages=["Production"])
            if versions:
                uri = f"models:/{name}/Production"
                logger.info(f"Production model: {name} v{versions[0].version}")
                return uri
        except Exception:
            pass

        logger.warning(f"No production model found for '{name}'")
        return None

    def get_latest_version(self, model_name: str | None = None) -> dict | None:
        """Get info about the latest model version.

        Returns:
            Dictionary with version info.
        """
        name = model_name or self.model_name
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                return {
                    "name": name,
                    "version": latest.version,
                    "stage": latest.current_stage,
                    "run_id": latest.run_id,
                    "status": latest.status,
                }
        except Exception as e:
            logger.error(f"Error fetching model versions: {e}")

        return None

    def list_models(self) -> list[dict]:
        """List all registered models and their versions.

        Returns:
            List of model info dictionaries.
        """
        models = []
        for rm in self.client.search_registered_models():
            versions = self.client.search_model_versions(f"name='{rm.name}'")
            models.append({
                "name": rm.name,
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                    }
                    for v in versions
                ],
            })
        return models
