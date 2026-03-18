"""
Data Ingestion Module
=====================
Handles loading raw datasets from local files or downloading from Kaggle.
Supports Telco, Bank, and E-commerce churn datasets.
"""

import pandas as pd
from loguru import logger

from src.utils.config import get_config, get_dataset_config, get_path
from src.utils.helpers import load_dataframe, timer


@timer
def ingest_dataset(dataset_name: str | None = None) -> pd.DataFrame:
    """Load a raw dataset from disk.

    Args:
        dataset_name: Name of dataset to load (telco, bank, ecommerce).
                      Uses active_dataset from config if None.

    Returns:
        Raw DataFrame loaded from the dataset file.
    """
    dataset_config = get_dataset_config(dataset_name)
    raw_dir = get_path("data_raw")
    filepath = raw_dir / dataset_config["filename"]

    if not filepath.exists():
        logger.warning(
            f"Dataset file not found: {filepath}. "
            f"Run `python scripts/download_data.py` to download it."
        )
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Download it from Kaggle: {dataset_config.get('kaggle_dataset', 'N/A')}"
        )

    df = load_dataframe(filepath)

    # Basic info logging
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Target column: {dataset_config['target']}")
    logger.info(f"Target distribution:\n{df[dataset_config['target']].value_counts()}")

    return df


@timer
def ingest_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all configured datasets.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    get_config()
    datasets = {}

    for dataset_name in ["telco", "bank", "ecommerce"]:
        try:
            datasets[dataset_name] = ingest_dataset(dataset_name)
        except FileNotFoundError:
            logger.warning(f"Skipping {dataset_name} - file not found")

    logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets


if __name__ == "__main__":
    # Run standalone: python -m src.data.ingest
    df = ingest_dataset()
    print(f"\nDataset loaded successfully: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
