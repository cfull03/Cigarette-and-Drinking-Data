from __future__ import annotations
from pathlib import Path
from typing import Dict
import yaml

def load_paths(paths_file: str | Path = "configs/paths.yaml") -> Dict[str, str]:
    """Load path config from YAML. Why: avoid hard-coded directories."""
    p = Path(paths_file)
    if not p.exists():
        raise FileNotFoundError(f"Paths file not found: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    required = {"raw_dir", "interim_dir", "processed_dir", "models_dir", "reports_dir"}
    missing = required - set(cfg)
    if missing:
        raise KeyError(f"Missing keys in {p}: {sorted(missing)}")
    return cfg 