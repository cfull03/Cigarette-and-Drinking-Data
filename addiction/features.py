# filepath: addiction/features.py
from __future__ import annotations

"""
Feature engineering utilities and CLI for the Cigarette & Drinking dataset.
"""

from dataclasses import dataclass, field
import functools
from pathlib import Path
from typing import Callable, Final, Iterable, List, Optional, Set, TypeVar

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from addiction.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Feature engineering CLI for the Cigarette & Drinking dataset.")

__all__ = [
    "FeatureError",
    "FeatureSpec",
    "FeatureRegistry",
    "REGISTRY",
    "build_features",
]

# -----------------------------
# Utilities
# -----------------------------
def _mode_safe(s: pd.Series) -> object:
    m = s.mode(dropna=True)
    return m.iat[0] if not m.empty else np.nan


def _has_all(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return set(cols).issubset(df.columns)


def _drop_target_cols(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop target column and any existing one-hot columns with '<target>_' prefix.
    Why: prevent leakage when fitting preprocessors; not for general feature files.
    """
    out = df.copy()
    pref = f"{target}_"
    to_drop = [c for c in out.columns if c == target or c.startswith(pref)]
    if to_drop:
        logger.info(f"Dropping target-derived columns: {to_drop}")
        out = out.drop(columns=to_drop, errors="ignore")
    return out


# -----------------------------
# FeatureSpec & Registry
# -----------------------------
class FeatureError(RuntimeError):
    """Registry or application error."""


@dataclass(order=True, frozen=True)
class FeatureSpec:
    order: int
    name: str = field(compare=False)
    func: Callable[[pd.DataFrame], pd.DataFrame] = field(compare=False)

    requires: Set[str] = field(default_factory=set, compare=False)
    produces: Set[str] = field(default_factory=set, compare=False)
    desc: str = field(default="", compare=False)
    enabled: bool = field(default=True, compare=False)

    def applicable(self, df: pd.DataFrame) -> bool:
        return _has_all(df, self.requires)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            logger.info(f"[disabled:{self.name}]")
            return df

        if not self.applicable(df):
            missing = self.requires - set(df.columns)
            logger.warning(f"[skip:{self.name}] Missing columns: {sorted(missing)}")
            return df

        try:
            out = self.func(df)
            if not isinstance(out, pd.DataFrame):
                raise FeatureError(f"Feature '{self.name}' must return a DataFrame.")
            return out
        except Exception as exc:
            logger.exception(f"[error:{self.name}] {exc}")
            return df


F = TypeVar("F", bound=Callable[[pd.DataFrame], pd.DataFrame])


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise FeatureError(f"Duplicate feature name: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def feature(
        self,
        name: str,
        *,
        requires: Iterable[str] = (),
        produces: Iterable[str] = (),
        order: int = 100,
        desc: str = "",
        enabled: bool = True,
        strict_produces: bool = False,
    ) -> Callable[[F], F]:
        reqs = set(requires)
        prods = set(produces)
        feature_name = name

        def deco(func: F) -> F:
            @functools.wraps(func)
            def wrapper(df: pd.DataFrame) -> pd.DataFrame:
                missing = reqs - set(df.columns)
                if missing:
                    raise AssertionError(
                        f"[{feature_name}] Missing required columns: {sorted(missing)}"
                    )

                out = func(df)
                if not isinstance(out, pd.DataFrame):
                    raise FeatureError(f"Feature '{feature_name}' must return a DataFrame.")

                if prods:
                    missing_out = prods - set(out.columns)
                    if missing_out:
                        msg = (
                            f"[{feature_name}] Declared 'produces' missing from output: "
                            f"{sorted(missing_out)}"
                        )
                        if strict_produces:
                            raise FeatureError(msg)
                        else:
                            logger.warning(msg)
                return out

            spec = FeatureSpec(
                name=feature_name,
                requires=reqs,
                produces=prods,
                func=wrapper,
                order=order,
                desc=desc,
                enabled=enabled,
            )
            self.register(spec)

            # Expose metadata on the callable for introspection/debugging.
            setattr(wrapper, "__feature_spec__", spec)
            setattr(wrapper, "__feature_name__", feature_name)
            setattr(wrapper, "__feature_requires__", tuple(sorted(reqs)))
            setattr(wrapper, "__feature_produces__", tuple(sorted(prods)))
            setattr(wrapper, "__original_func__", func)

            return wrapper
        return deco

    def names(self) -> list[str]:
        return sorted(self._specs.keys())

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise FeatureError(f"Unknown feature: {name}") from exc

    def build(
        self,
        df: pd.DataFrame,
        *,
        only: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        specs = list(self._specs.values())
        specs.sort()

        if only:
            missing = only - set(self._specs)
            if missing:
                raise FeatureError(f"`only` includes unknown features: {sorted(missing)}")
            specs = [s for s in specs if s.name in only]
        if exclude:
            unknown = exclude - set(self._specs)
            if unknown:
                raise FeatureError(f"`exclude` includes unknown features: {sorted(unknown)}")
            specs = [s for s in specs if s.name not in exclude]

        out = df.copy()
        for spec in specs:
            out = spec.apply(out)
        return out


REGISTRY: Final[FeatureRegistry] = FeatureRegistry()

# -----------------------------
# Feature implementations
# -----------------------------
@REGISTRY.feature(
    name="basic_cleanup",
    requires=(),
    produces=(),
    order=0,
    desc="Drop PII: remove 'id' and 'name' columns if present.",
)
def feat_basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_cols = [c for c in ("id", "name") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


@REGISTRY.feature(
    name="income_features",
    requires=("annual_income_usd",),
    produces=("income_band", "log_income", "income_z", "income_decile"),
    order=10,
    desc="Bands, log, z-score, and deciles from annual income.",
)
def feat_income(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inc = pd.to_numeric(out["annual_income_usd"], errors="coerce")

    bins = [-np.inf, 25_000, 50_000, 75_000, 100_000, 150_000, np.inf]
    labels = ["<25k", "25–50k", "50–75k", "75–100k", "100–150k", "150k+"]
    out["income_band"] = pd.cut(inc, bins=bins, labels=labels, include_lowest=True, right=True)

    out["log_income"] = np.log1p(inc)
    mu, sigma = inc.mean(), inc.std()
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0
    out["income_z"] = (inc - mu) / sigma
    try:
        out["income_decile"] = pd.qcut(inc, 10, labels=False, duplicates="drop")
    except Exception:
        out["income_decile"] = np.nan
    return out


@REGISTRY.feature(
    name="smoke_intensity",
    requires=("smokes_per_day",),
    produces=("smoke_intensity",),
    order=20,
    desc="Bucketize smoking intensity.",
)
def feat_smoke_intensity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["smoke_intensity"] = pd.cut(
        pd.to_numeric(out["smokes_per_day"], errors="coerce"),
        [0, 1, 5, 10, 20, np.inf],
        labels=["none", "ultra", "light", "med", "heavy"],
        include_lowest=True,
    )
    return out


@REGISTRY.feature(
    name="drink_intensity",
    requires=("drinks_per_week",),
    produces=("drink_intensity",),
    order=20,
    desc="Bucketize drinking intensity.",
)
def feat_drink_intensity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drink_intensity"] = pd.cut(
        pd.to_numeric(out["drinks_per_week"], errors="coerce"),
        [0, 1, 7, 14, 28, np.inf],
        labels=["none", "very_low", "low", "mod", "high"],
        include_lowest=True,
    )
    return out


@REGISTRY.feature(
    name="ratios_dependents",
    requires=("children_count", "annual_income_usd"),
    produces=("dependents_ratio",),
    order=30,
    desc="(children+1)/(income+1) as affordability proxy.",
)
def feat_dependents_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    kids = pd.to_numeric(out["children_count"], errors="coerce").fillna(0)
    inc = pd.to_numeric(out["annual_income_usd"], errors="coerce").fillna(0)
    out["dependents_ratio"] = (kids + 1) / (inc + 1)
    return out


@REGISTRY.feature(
    name="quit_effort_smoke_norm",
    requires=("attempts_to_quit_smoking", "smokes_per_day"),
    produces=("quit_effort_smoke_norm",),
    order=30,
    desc="Normalized smoking quit attempts.",
)
def feat_quit_smoke(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    attempts = pd.to_numeric(out["attempts_to_quit_smoking"], errors="coerce").fillna(0)
    per_day = pd.to_numeric(out["smokes_per_day"], errors="coerce").fillna(0)
    out["quit_effort_smoke_norm"] = attempts / (per_day + 1)
    return out


@REGISTRY.feature(
    name="quit_effort_drink_norm",
    requires=("attempts_to_quit_drinking", "drinks_per_week"),
    produces=("quit_effort_drink_norm",),
    order=30,
    desc="Normalized drinking quit attempts.",
)
def feat_quit_drink(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    attempts = pd.to_numeric(out["attempts_to_quit_drinking"], errors="coerce").fillna(0)
    per_week = pd.to_numeric(out["drinks_per_week"], errors="coerce").fillna(0)
    out["quit_effort_drink_norm"] = attempts / (per_week + 1)
    return out


@REGISTRY.feature(
    name="impute_social_support",
    requires=("social_support", "marital_status", "children_count"),
    produces=("social_support",),
    order=40,
    desc="Groupwise mode imputation by (marital_status, children_count).",
)
def feat_impute_social(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(["marital_status", "children_count"])["social_support"].transform(_mode_safe)
    glob = _mode_safe(out["social_support"])
    out["social_support"] = out["social_support"].fillna(grp).fillna(glob)
    return out


@REGISTRY.feature(
    name="impute_education_level",
    requires=("education_level", "employment_status", "income_band"),
    produces=("education_level",),
    order=40,
    desc="Groupwise mode imputation by (employment_status, income_band).",
)
def feat_impute_edu(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(["employment_status", "income_band"])["education_level"].transform(_mode_safe)
    glob = _mode_safe(out["education_level"])
    out["education_level"] = out["education_level"].fillna(grp).fillna(glob)
    return out


@REGISTRY.feature(
    name="impute_therapy_history",
    requires=("therapy_history", "education_level", "marital_status", "mental_health_status"),
    produces=("therapy_history",),
    order=40,
    desc="Groupwise mode imputation by (education_level, marital_status, mental_health_status).",
)
def feat_impute_therapy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = (
        out.groupby(["education_level", "marital_status", "mental_health_status"])["therapy_history"]
           .transform(_mode_safe)
    )
    glob = _mode_safe(out["therapy_history"])
    out["therapy_history"] = out["therapy_history"].fillna(grp).fillna(glob)
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_features(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """
    Build domain features via the FeatureSpec registry.
    NOTE: target is not dropped here. Use CLI --drop-target if you need X-only.
    """
    if target is not None:
        logger.debug("build_features(target=...) is ignored; target is not dropped here.")
    return REGISTRY.build(df)


__all__ = [
    "FeatureError",
    "FeatureSpec",
    "FeatureRegistry",
    "REGISTRY",
    "build_features",
]


# -----------------------------
# CLI
# -----------------------------
@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    only: Optional[str] = typer.Option(
        None,
        help="Comma-separated feature names to run exclusively.",
    ),
    exclude: Optional[str] = typer.Option(
        None,
        help="Comma-separated feature names to skip.",
    ),
    target: Optional[str] = typer.Option(
        None,
        help="Target column name (used only when --drop-target is set).",
    ),
    drop_target: bool = typer.Option(
        False,
        help="If set, drop target and any '<target>_*' columns before building features.",
    ),
    list_features: bool = typer.Option(
        False, "--list", "-l", help="List available features (by order) and exit."
    ),
) -> None:
    """Load CSV, optionally drop target(+OHE) for X-only, build features, write CSV."""
    if list_features:
        specs: List[FeatureSpec] = sorted(REGISTRY._specs.values())  # type: ignore[attr-defined]
        for s in specs:
            status = "ENABLED" if s.enabled else "DISABLED"
            print(f"{s.order:>3} | {s.name:<24} | {status:<8} | requires={sorted(s.requires)}")
        raise typer.Exit(code=0)

    if not input_path.exists():
        typer.echo(f"[ERROR] Input not found: {input_path}")
        raise typer.Exit(code=1)

    only_set = set(map(str.strip, only.split(","))) if only else None
    exclude_set = set(map(str.strip, exclude.split(","))) if exclude else None

    logger.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Drop only when explicitly requested
    if drop_target:
        if not target:
            typer.echo("[ERROR] --drop-target requires --target <name>")
            raise typer.Exit(code=2)
        df = _drop_target_cols(df, target)
    elif target:
        logger.warning("--target provided without --drop-target; leaving target in place.")

    logger.info("Building features…")
    for _ in tqdm(range(1), total=1):
        df_feat = REGISTRY.build(df, only=only_set, exclude=exclude_set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(output_path, index=False)
    logger.success(f"Wrote features → {output_path}")


if __name__ == "__main__":
    app()
