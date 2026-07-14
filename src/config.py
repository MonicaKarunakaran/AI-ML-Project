"""Project configuration settings."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
