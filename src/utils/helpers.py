"""
Common helper functions used across the pipeline.
"""

import time
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def timer(func):
    """Decorator to log execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱️  {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def load_dataframe(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet based on file extension.

    Args:
        path: Path to the file.
        **kwargs: Additional arguments passed to the reader.

    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path, **kwargs)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"📂 Loaded {path.name}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save a DataFrame to CSV or Parquet based on file extension.

    Args:
        df: DataFrame to save.
        path: Output path.
        **kwargs: Additional arguments passed to the writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif path.suffix in (".parquet", ".pq"):
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"💾 Saved {path.name}: {df.shape[0]} rows × {df.shape[1]} cols")


def get_class_distribution(y: pd.Series | np.ndarray) -> dict[str, Any]:
    """Get class distribution statistics.

    Args:
        y: Target variable.

    Returns:
        Dictionary with class counts and ratios.
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    counts = y.value_counts()
    total = len(y)

    return {
        "total_samples": total,
        "class_counts": counts.to_dict(),
        "class_ratios": (counts / total).to_dict(),
        "imbalance_ratio": counts.min() / counts.max(),
    }


def create_gitkeep_files(base_dir: str | Path) -> None:
    """Create .gitkeep files in empty directories to preserve them in Git."""
    base_dir = Path(base_dir)
    for dir_path in base_dir.rglob("*"):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            (dir_path / ".gitkeep").touch()
            logger.debug(f"Created .gitkeep in {dir_path}")
