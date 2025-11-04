"""
Project-wide configuration and paths for the CCDS layout.

- Loads environment variables from .env if present.
- Defines canonical project directories (data, models, reports, etc.).
- Configures Loguru; integrates with tqdm if available.
- Provides a helper to create the standard directory tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from loguru import logger

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
# Load environment variables from .env file if it exists (no error if missing)
load_dotenv()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Resolve project root as the parent of this file's directory (i.e., repo root)
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
logger.debug(f"Resolved PROJ_ROOT: {PROJ_ROOT}")

# Data directories (CCDS-style)
DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# Models
MODELS_DIR: Path = PROJ_ROOT / "models"

# Reports / figures
REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# -----------------------------------------------------------------------------
# Logging (Loguru + tqdm-friendly sink if tqdm is installed)
# -----------------------------------------------------------------------------
def _configure_logging() -> None:
    """
    Configure loguru to play nicely with tqdm progress bars if available.
    Falls back to the default sink otherwise.
    """
    try:
        # Defer import so tqdm isn't a hard dependency
        from tqdm import tqdm  # type: ignore

        # Remove default sink (id=0) and add a tqdm-aware sink
        try:
            logger.remove(0)
        except Exception:
            # In case the default sink id differs (e.g., re-imported)
            logger.remove()

        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        logger.debug("Configured loguru with tqdm sink.")
    except ModuleNotFoundError:
        # tqdm not installed; keep default logging configuration
        logger.debug("tqdm not found; using default loguru sink.")


_configure_logging()

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
    # Helpers
    "ensure_project_dirs",
]
