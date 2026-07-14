"""Utility helper Functions."""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
