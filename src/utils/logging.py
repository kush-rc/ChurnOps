"""
Structured logging setup using Loguru.

Provides consistent, structured logging across the entire pipeline
with JSON formatting for production and pretty formatting for development.
"""

import sys

from loguru import logger

from src.utils.config import PROJECT_ROOT


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
) -> None:
    """Configure structured logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (relative to project root)
        json_format: If True, use JSON format for log output
    """
    # Remove default handler
    logger.remove()

    # Console handler (always pretty)
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (if specified)
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            serialize=json_format,
        )

    logger.info(f"Logging configured: level={level}, file={log_file}")


# Initialize default logging on import
setup_logging()
