"""
Script to export the latest production model from MLflow into a local bundled folder.
This enables deploying the model without requiring a live connection to an MLflow tracking server.
"""

import os
import shutil

import mlflow
from loguru import logger
from src.models.registry import ModelRegistry
from src.utils.config import get_config


def export_production_model(domain: str) -> None:
    """Export the production model from MLflow to a local folder for bundling."""
    config = get_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    registry = ModelRegistry()
    
    # Check if there is a production model
    model_uri = registry.get_production_model_uri(f"{domain}-churn-model")
    if not model_uri:
        logger.error(f"No production model found for domain: {domain}")
        return

    logger.info(f"Found production model: {model_uri}")

    # Destination path
    dest_path = f"models/production/{domain}"
    
    # Remove old bundle if it exists
    if os.path.exists(dest_path):
        logger.info(f"Removing old model bundle at {dest_path}")
        shutil.rmtree(dest_path)
    
    logger.info(f"Downloading model artifacts to {dest_path}...")
    
    # Download the MLflow model to a temp local dir, then move it to dest_path
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    shutil.copytree(local_path, dest_path)
    
    logger.info(f"Successfully bundled production model into {dest_path}")


if __name__ == "__main__":
    export_production_model("telco")
