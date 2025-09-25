# file: src/addiction/utilities/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


@dataclass(frozen=True)
class ConfigPath:
    raw_csv: Path
    processed_dir: Path
    interim_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    @staticmethod
    def _resolve(root: Path, p: Optional[str], default: str, is_file: bool = False) -> Path:
        base = Path(p or default)
        return base if base.is_absolute() else (root / base).resolve()

    @classmethod
    def from_mapping(cls, paths: Mapping[str, Any], root: Path) -> "ConfigPath":
        return cls(
            raw_csv=cls._resolve(root, paths.get("raw_csv"), "data/raw/addiction_population_data.csv", is_file=True),
            processed_dir=cls._resolve(root, paths.get("processed_dir"), "data/processed"),
            interim_dir=cls._resolve(root, paths.get("interim_dir"), "data/interim"),
            models_dir=cls._resolve(root, paths.get("models_dir"), "models"),
            reports_dir=cls._resolve(root, paths.get("reports_dir"), "reports"),
            figures_dir=cls._resolve(root, paths.get("figures_dir"), "reports/figures"),
        )

    def ensure_dirs(self, include_figures: bool = True) -> None:
        for p in self.iter_dirs(include_figures=include_figures):
            p.mkdir(parents=True, exist_ok=True)

    def iter_dirs(self, include_figures: bool = True):
        yield self.processed_dir
        yield self.interim_dir
        yield self.models_dir
        yield self.reports_dir
        if include_figures:
            yield self.figures_dir


@dataclass(frozen=True)
class Config:
    _raw: Dict[str, Any]
    _root: Path
    paths: ConfigPath

    @property
    def root(self) -> Path:
        return self._root

    @property
    def d(self) -> Dict[str, Any]:
        return self._raw

    def get(self, key: str, default: Any = None) -> Any:
        cur: Any = self._raw
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def require(self, key: str) -> Any:
        sentinel = object()
        cur: Any = self._raw
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                cur = sentinel
                break
        if cur is sentinel:
            raise KeyError(f"Missing required config key: '{key}'")
        return cur

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "Config":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        root = cls._detect_root(cfg_path.parent)

        raw.setdefault("project", {})
        raw.setdefault("paths", {})
        raw.setdefault("training", {})
        raw.setdefault("evaluation", {})
        raw.setdefault("artifacts", {})

        def interp(obj: Any) -> Any:
            if isinstance(obj, str):
                for k, v in os.environ.items():
                    obj = obj.replace(f"${{{k}}}", v)
                return obj
            if isinstance(obj, Mapping):
                return {k: interp(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [interp(v) for v in obj]
            return obj

        raw = interp(raw)
        paths = ConfigPath.from_mapping(raw.get("paths", {}), root=root)
        return cls(_raw=raw, _root=root, paths=paths)

    @staticmethod
    def _detect_root(start: Path) -> Path:
        cur = start.resolve()
        for _ in range(6):
            if (cur / ".git").exists() or (cur / "pyproject.toml").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return start.resolve()


# ---- convenience function (import target) ----
def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Preferred import for other modules."""
    return Config.load(path)
