"""API dependency injection."""

from functools import lru_cache

from src.utils.config import get_config


@lru_cache()
def get_settings():
    """Get cached application settings."""
    return get_config()
