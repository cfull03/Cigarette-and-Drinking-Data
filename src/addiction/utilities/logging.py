# file: src/addiction/utilities/logging.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional, Union
from datetime import datetime

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

from .config import Config, load_config

_CONFIGURED = False
_FILE_HANDLER: Optional[logging.Handler] = None
_CONSOLE_HANDLER: Optional[logging.Handler] = None


# ---- internals ----

def _level(level: Optional[Union[int, str]]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def _log_dir(cfg: Config) -> Path:
    # Why: keep logs grouped; avoids cluttering reports root
    d = cfg.paths.reports_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_handlers(cfg: Config, level: int, to_file: bool) -> None:
    global _CONFIGURED, _FILE_HANDLER, _CONSOLE_HANDLER
    if _CONFIGURED:
        # Update levels on existing handlers for consistency across calls
        if _CONSOLE_HANDLER:
            _CONSOLE_HANDLER.setLevel(level)
        if _FILE_HANDLER:
            _FILE_HANDLER.setLevel(level)
        logging.getLogger().setLevel(level)
        return

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console (pretty)
    if _HAS_RICH:
        _CONSOLE_HANDLER = RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_level=True, show_path=False)
        fmt = logging.Formatter("%(message)s")
    else:
        _CONSOLE_HANDLER = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _CONSOLE_HANDLER.setLevel(level)
    _CONSOLE_HANDLER.setFormatter(fmt)
    root.addHandler(_CONSOLE_HANDLER)

    # Optional file
    if to_file:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run = str(cfg.get("project.run_name", "run"))
        logfile = _log_dir(cfg) / f"{run}_{ts}.log"
        _FILE_HANDLER = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        _FILE_HANDLER.setLevel(level)
        _FILE_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        root.addHandler(_FILE_HANDLER)

    _CONFIGURED = True


# ---- public API ----

def get_logger(
    name: str = "addiction",
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    level: Optional[Union[int, str]] = None,
    to_file: bool = True,
) -> logging.Logger:
    """
    Returns a configured logger. Safe to call multiple times.
    """
    cfg_obj = cfg if isinstance(cfg, Config) else load_config(cfg)
    lvl = _level(level if level is not None else cfg_obj.get("logging.level", "INFO"))
    _setup_handlers(cfg_obj, lvl, to_file=to_file)
    return logging.getLogger(name)


def set_verbosity(level: Union[int, str]) -> None:
    """
    Dynamically adjust verbosity across all handlers.
    """
    lvl = _level(level)
    logging.getLogger().setLevel(lvl)
    for h in logging.getLogger().handlers:
        h.setLevel(lvl)


def log_exceptions(logger: Optional[logging.Logger] = None) -> Callable[[Callable[..., Any]], Callable[..., int]]:
    """
    Decorator for CLI entrypoints: logs exceptions and returns exit code 1.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., int]:
        log = logger or get_logger(fn.__module__)
        def wrapper(*args, **kwargs) -> int:
            try:
                fn(*args, **kwargs)
                return 0
            except SystemExit as se:
                # Why: preserve intended exit codes
                code = int(getattr(se, "code", 1) or 0)
                if code != 0:
                    log.exception("Exited with error code %s", code)
                return code
            except Exception:
                log.exception("Unhandled exception")
                return 1
        return wrapper
    return deco
