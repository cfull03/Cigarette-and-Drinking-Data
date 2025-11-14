# filepath: addiction/config.py
# [exp-001] - Contains methods modified/added in exp/001-smoking-trends-cf
"""
Project-wide configuration and paths for the CCDS layout.

- Loads environment variables from .env if present.
- Defines canonical project directories (data, models, reports, etc.).
- Configures Loguru; integrates with tqdm if available.
- Provides a helper to create the standard directory tree.
- Exposes commonly used config variables (target, split size, encoders, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from loguru import logger

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
# Load environment variables from .env file if it exists (no error if missing)
load_dotenv()

# Small parsers for robust env handling
def _getenv_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default
    # [exp-001]

def _getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        logger.warning(f"Invalid int for {name}={v!r}; using default {default}")
        return default
    # [exp-001]

def _getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        logger.warning(f"Invalid float for {name}={v!r}; using default {default}")
        return default
    # [exp-001]

def _getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    # [exp-001]

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Resolve project root as the parent of this file's directory (i.e., repo root)
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
logger.debug(f"Resolved PROJ_ROOT: {PROJ_ROOT}")

# Allow overriding DATA_DIR via env (e.g., for ephemeral runs)
_DATA_DIR_ENV = _getenv_str("DATA_DIR", str(PROJ_ROOT / "data"))
DATA_DIR: Path = Path(_DATA_DIR_ENV)

# Data directories (CCDS-style)
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# Models
MODELS_DIR: Path = Path(_getenv_str("MODELS_DIR", str(PROJ_ROOT / "models")))

# Reports / figures
REPORTS_DIR: Path = Path(_getenv_str("REPORTS_DIR", str(PROJ_ROOT / "reports")))
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Common file names (overridable)
DATASET_RAW_NAME: str = _getenv_str("DATASET_RAW_NAME", "addiction_population_data.csv")
RAW_DATA_DEFAULT_PATH: Path = RAW_DATA_DIR / DATASET_RAW_NAME

# -----------------------------------------------------------------------------
# Logging (Loguru + tqdm-friendly sink if tqdm is installed)
# -----------------------------------------------------------------------------
def _configure_logging() -> None:
    """
    Configure loguru to play nicely with tqdm progress bars if available.
    Falls back to the default sink otherwise.
    """
    # Set log level from env (default INFO)
    log_level = _getenv_str("LOG_LEVEL", "INFO").upper()
    try:
        logger.remove()
    except Exception:
        pass
    # Try tqdm-integrated sink first
    try:
        from tqdm import tqdm  # type: ignore

        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=log_level)
        logger.debug("Configured loguru with tqdm sink.")
    except ModuleNotFoundError:
        # tqdm not installed; standard stderr sink
        logger.add(lambda msg: print(msg, end=""), colorize=True, level=log_level)
        logger.debug("tqdm not found; using default loguru sink.")

_configure_logging()

# -----------------------------------------------------------------------------
# Project-wide knobs (loaded from env with safe defaults)
# -----------------------------------------------------------------------------
# Modeling / data split
TARGET: str = _getenv_str("TARGET", "has_health_issues")
TEST_SIZE: float = _getenv_float("TEST_SIZE", 0.2)
RANDOM_STATE: int = _getenv_int("RANDOM_STATE", 42)  # alias for sklearn split
RANDOM_SEED: int = _getenv_int("RANDOM_SEED", RANDOM_STATE)  # general-purpose seed

# Preprocessing
ENCODE_CATEGORICALS: bool = _getenv_bool("ENCODE_CATEGORICALS", True)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_project_dirs(extra: Iterable[Path] | None = None) -> None:
    """
    Create the standard CCDS directory tree if missing.

    Parameters
    ----------
    extra : Iterable[Path] | None
        Any additional directories you want created.
    """
    default_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    if extra:
        default_dirs.extend(list(extra))

    for p in default_dirs:
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {p}")

    logger.info("Project directories are ready.")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Paths
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "DATASET_RAW_NAME",
    "RAW_DATA_DEFAULT_PATH",
    # Knobs
    "TARGET",
    "TEST_SIZE",
    "RANDOM_STATE",
    "RANDOM_SEED",
    "ENCODE_CATEGORICALS",
    # Helpers
    "ensure_project_dirs",
]
