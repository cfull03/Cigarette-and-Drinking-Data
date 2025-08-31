#!/usr/bin/env python3
"""scripts/sample.py

Generate a sample dataset using the project's IO helpers.
- Defaults to the `addiction_ds` package and `configs/default.yaml`.
- Picks `raw` if present else falls back to `sample_input`.
- Prints the output sample path for Make/logs.

Keep this script thin and delegate to `src/addiction_ds/io.py`.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample dataset")
    parser.add_argument(
        "--module",
        default=os.environ.get("MODULE", "addiction_ds"),
        help="Top-level package exposing an `io` module (default: addiction_ds)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG", "configs/default.yaml"),
        help="Path to the YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(os.environ.get("N", 100)),
        help="Rows to sample (default: 100)",
    )
    return parser.parse_args(argv)


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        io = importlib.import_module(f"{args.module}.io")
    except Exception as exc:
        print(f"Failed to import '{args.module}.io': {exc}", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    try:
        _ensure_exists(cfg_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        cfg: Any = io.load_cfg(str(cfg_path))
        P: dict[str, Any] = io.get_paths(cfg)

        raw = P.get("raw")
        try:
            src = raw if (raw is not None and raw.exists()) else P["sample_input"]
        except Exception:
            src = P["sample_input"]

        print(f"Loading: {src}")
        df = io.read_csv(src)
        out = io.write_sample(df, cfg, n=args.n, index=False)
        print("sample ->", out)
        return 0
    except Exception as exc:
        print(f"Error generating sample: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
