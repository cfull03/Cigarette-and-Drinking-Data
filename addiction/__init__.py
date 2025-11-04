"""
addiction
=========

Lightweight top-level API for the project package.

What you get on import:
- Configured logger (from `config.py`, tqdm-aware if available)
- Canonical CCDS paths (DATA_DIR, RAW_DATA_DIR, etc.)
- `ensure_project_dirs()` helper
- Package `__version__`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

# Expose the configured loguru logger (same instance configured in config)
from loguru import logger as log  # noqa: F401

# Import config first so logging/paths are initialized once here.
from . import config as _config  # noqa: F401

# Re-export commonly used symbols from config
from .config import (  # noqa: F401
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
)

# Package version (best-effort)
try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import PackageNotFoundError, version  # type: ignore
    except Exception:  # pragma: no cover
        version = None  # type: ignore
        PackageNotFoundError = Exception  # type: ignore


def _get_version() -> str:
    if version is None:
        return "0.0.0"
    try:
        return version("addiction")
    except PackageNotFoundError:
        # Editable installs before build metadata exists, or running from source
        return "0.0.0"


__version__ = _get_version()


def setup(extra_dirs: Optional[Iterable[Path]] = None) -> None:
    """
    One-call convenience to ensure the CCDS directory tree exists.

    Parameters
    ----------
    extra_dirs : Optional[Iterable[pathlib.Path]]
        Any additional directories to create.

    Examples
    --------
    >>> from addiction import setup
    >>> setup()  # creates data/, reports/, models/, etc.
    """
    ensure_project_dirs(extra=extra_dirs)
    log.info("addiction setup complete.")


def __repr__() -> str:  # pragma: no cover
    return f"<addiction {__version__} root={PROJ_ROOT}>"

__all__ = [
    # version
    "__version__",
    # logger
    "log",
    # paths
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    # helpers
    "ensure_project_dirs",
    "setup",
]

