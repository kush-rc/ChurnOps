"""
Configuration loader for the Churn Prediction MLOps pipeline.

Loads YAML configuration files and provides typed access
to project settings.
"""

from pathlib import Path

import yaml
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    """Load the main project configuration."""
    return load_yaml(CONFIGS_DIR / "config.yaml")


def get_dataset_config(dataset_name: str | None = None) -> dict:
    """Load dataset-specific configuration.

    Args:
        dataset_name: Name of the dataset (telco, bank, ecommerce).
                      If None, uses the active_dataset from main config.
    """
    if dataset_name is None:
        main_config = get_config()
        dataset_name = main_config.get("active_dataset", "telco")

    config_path = CONFIGS_DIR / "data" / f"{dataset_name}.yaml"
    config = load_yaml(config_path)
    logger.info(f"Loaded dataset config: {dataset_name}")
    return config["dataset"]


def get_model_config(model_name: str) -> dict:
    """Load model-specific configuration.

    Args:
        model_name: Name of the model (xgboost, lightgbm, etc.)
    """
    config_path = CONFIGS_DIR / "model" / f"{model_name}.yaml"
    config = load_yaml(config_path)
    logger.info(f"Loaded model config: {model_name}")
    return config["model"]


def get_training_config() -> dict:
    """Load training configuration."""
    config = load_yaml(CONFIGS_DIR / "training" / "training.yaml")
    return config["training"]


def get_monitoring_config() -> dict:
    """Load monitoring/drift detection configuration."""
    config = load_yaml(CONFIGS_DIR / "monitoring" / "drift.yaml")
    return config["monitoring"]


def get_path(key: str) -> Path:
    """Get a configured path, resolved relative to project root.

    Args:
        key: Path key from config (e.g., 'data_raw', 'models')
    """
    config = get_config()
    relative_path = config["paths"].get(key)
    if relative_path is None:
        raise KeyError(f"Path key '{key}' not found in config")
    path = PROJECT_ROOT / relative_path
    path.mkdir(parents=True, exist_ok=True)
    return path
