# file: src/addiction/data/ingest.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utilities.config import Config, load_config
from ..utilities.io import save_json
from ..utilities.logging import get_logger


# ----------------------------- schema structures -----------------------------
@dataclass(frozen=True)
class FieldRule:
    name: str
    type: str                 # "integer" | "number" | "boolean" | "string"
    nullable: bool = True
    enum: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None

@dataclass(frozen=True)
class DatasetSchema:
    primary_key: List[str]
    fields: Dict[str, FieldRule]


# ----------------------------- schema loading -----------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_schema(schema_path: str | Path = "config/schema.yaml") -> DatasetSchema:
    data = _read_yaml(Path(schema_path))
    fields: Dict[str, FieldRule] = {}
    for f in data.get("fields", []):
        stats = (f.get("stats") or {}) if isinstance(f.get("stats"), dict) else {}
        fields[f["name"]] = FieldRule(
            name=f["name"],
            type=str(f.get("type", "string")).lower(),
            nullable=bool(f.get("nullable", True)),
            enum=f.get("enum"),
            min=stats.get("min", f.get("min")),
            max=stats.get("max", f.get("max")),
        )
    return DatasetSchema(primary_key=list(data.get("primary_key", [])), fields=fields)


# ----------------------------- cleaning & coercion -----------------------------
def _coerce_column(s: pd.Series, rule: FieldRule) -> pd.Series:
    if rule.type == "integer":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if rule.type == "number":
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if rule.type == "boolean":
        true = {"true","1","yes","y","t"}; false = {"false","0","no","n","f"}
        def to_bool(x):
            if pd.isna(x): return pd.NA
            if isinstance(x, (bool, pd.BooleanDtype.type)): return x
            xs = str(x).strip().lower()
            if xs in true: return True
            if xs in false: return False
            return pd.NA
        return s.map(to_bool).astype("boolean")
    return s.astype("string")


@dataclass
class CleanReport:
    input_rows: int
    output_rows: int
    rejects_rows: int
    dropped_nullability: int
    dropped_enum: int
    dropped_range: int
    dropped_pk_dupes: int


def clean_dataframe(df: pd.DataFrame, schema: DatasetSchema) -> Tuple[pd.DataFrame, pd.DataFrame, CleanReport]:
    original_rows = len(df)
    want_cols = list(schema.fields.keys())
    for c in want_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[want_cols].copy()
    for name, rule in schema.fields.items():
        df[name] = _coerce_column(df[name], rule)

    valid = pd.Series(True, index=df.index); reasons: Dict[int, List[str]] = {}
    dropped_nullability = dropped_enum = dropped_range = dropped_pk_dupes = 0

    for name, rule in schema.fields.items():
        if not rule.nullable:
            bad = df[name].isna()
            if bad.any():
                dropped_nullability += int(bad.sum()); valid &= ~bad
                for i in df.index[bad]: reasons.setdefault(i, []).append(f"nullability:{name}")

    for name, rule in schema.fields.items():
        if rule.enum is not None:
            allowed = set(str(v) for v in rule.enum)
            bad = (~df[name].isna()) & (~df[name].astype("string").isin(allowed))
            if bad.any():
                dropped_enum += int(bad.sum()); valid &= ~bad
                for i in df.index[bad]: reasons.setdefault(i, []).append(f"enum:{name}")

    for name, rule in schema.fields.items():
        if rule.type in {"integer", "number"}:
            bad = pd.Series(False, index=df.index)
            if rule.min is not None: bad |= (~df[name].isna()) & (df[name] < rule.min)
            if rule.max is not None: bad |= (~df[name].isna()) & (df[name] > rule.max)
            if bad.any():
                dropped_range += int(bad.sum()); valid &= ~bad
                for i in df.index[bad]: reasons.setdefault(i, []).append(f"range:{name}")

    cleaned = df[valid].copy()
    rejects = df[~valid].copy()

    if schema.primary_key and not cleaned.empty:
        keep = cleaned.drop_duplicates(subset=schema.primary_key, keep="first")
        dropped_pk_dupes = len(cleaned) - len(keep)
        dups = cleaned[cleaned.duplicated(subset=schema.primary_key, keep="first")]
        if not dups.empty:
            dups = dups.copy(); dups["reject_reasons"] = "pk_duplicate"
            rejects = pd.concat([rejects, dups], axis=0, ignore_index=True)
        cleaned = keep

    if not rejects.empty and "reject_reasons" not in rejects.columns:
        rr = [";".join(reasons.get(i, ["unknown"])) for i in rejects.index]
        rejects = rejects.copy(); rejects["reject_reasons"] = rr

    report = CleanReport(
        input_rows=original_rows,
        output_rows=int(len(cleaned)),
        rejects_rows=int(len(rejects)),
        dropped_nullability=dropped_nullability,
        dropped_enum=dropped_enum,
        dropped_range=dropped_range,
        dropped_pk_dupes=dropped_pk_dupes,
    )
    return cleaned.reset_index(drop=True), rejects.reset_index(drop=True), report


# ----------------------------- transform stage (config-driven) -----------------------------
@dataclass(frozen=True)
class TransformConfig:
    outlier_mode: str = "iqr"        # "off" | "iqr"
    iqr_k: float = 1.5
    save_csv: bool = False
    save_parquet: bool = True

def _read_preproc(cfg: Config):
    p = cfg.get("preprocessing", {}) or {}
    num = p.get("numeric", {}); cat = p.get("categorical", {})
    return {
        "num_imputer_strategy": (num.get("imputer", {}) or {}).get("strategy", "median"),
        "num_standardize": (num.get("scaler", {}) or {}).get("standardize", True),
        "num_with_mean": (num.get("scaler", {}) or {}).get("with_mean", False),
        "cat_imputer_strategy": (cat.get("imputer", {}) or {}).get("strategy", "most_frequent"),
        "ohe_handle_unknown": (cat.get("one_hot", {}) or {}).get("handle_unknown", "ignore"),
        "ohe_sparse_output": (cat.get("one_hot", {}) or {}).get("sparse_output", True),
    }

def _pick_cols(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    drop_cols = set(cfg.get("data.drop_columns", []) or [])
    cols = [c for c in df.columns if c not in drop_cols]
    exp_num = set(cfg.get("data.numeric_columns", []) or [])
    exp_cat = set(cfg.get("data.categorical_columns", []) or [])
    if exp_num or exp_cat:
        num_cols = sorted([c for c in cols if c in exp_num])
        cat_cols = sorted([c for c in cols if c in exp_cat and c not in num_cols])
        if not num_cols: num_cols = sorted([c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in cat_cols])
        if not cat_cols: cat_cols = sorted([c for c in cols if c not in num_cols])
        return num_cols, cat_cols
    cat_cols = sorted([c for c in cols if not pd.api.types.is_numeric_dtype(df[c])])
    num_cols = sorted([c for c in cols if c not in cat_cols])
    return num_cols, cat_cols

def build_preprocessor_from_config(df: pd.DataFrame, cfg: Config) -> Tuple[ColumnTransformer, List[str], List[str]]:
    s = _read_preproc(cfg)
    num_cols, cat_cols = _pick_cols(df, cfg)

    num_steps = [("imputer", SimpleImputer(strategy=s["num_imputer_strategy"]))]
    if s["num_standardize"]:
        num_steps.append(("scaler", StandardScaler(with_mean=s["num_with_mean"])))
    num_pipe = Pipeline(num_steps)

    ohe_kwargs: Dict[str, Any] = {"handle_unknown": s["ohe_handle_unknown"]}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        ohe_kwargs["sparse_output"] = s["ohe_sparse_output"]
    else:
        ohe_kwargs["sparse"] = s["ohe_sparse_output"]
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=s["cat_imputer_strategy"])),
        ("ohe", OneHotEncoder(**ohe_kwargs)),
    ])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, num_cols, cat_cols

def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], k: float = 1.5) -> pd.DataFrame:
    if not numeric_cols: return df
    q1 = df[numeric_cols].quantile(0.25); q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1; lower = q1 - k * iqr; upper = q3 + k * iqr
    mask = ((df[numeric_cols] >= lower) & (df[numeric_cols] <= upper)).all(axis=1)
    return df[mask]


# ----------------------------- ingest -----------------------------
def _timestamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def process_one(
    csv_path: Path,
    cfg: Config,
    schema: DatasetSchema,
    logger,
    tcfg: TransformConfig = TransformConfig(),
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False)
    cleaned, rejects, rep = clean_dataframe(df, schema)

    base = cleaned
    num_for_outlier = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    if tcfg.outlier_mode == "iqr" and num_for_outlier:
        base = remove_outliers_iqr(base, num_for_outlier, k=tcfg.iqr_k)

    pre, _, cat_cols = build_preprocessor_from_config(base, cfg)
    X = pre.fit_transform(base)
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        cat_features = pre.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols).tolist()
        num_cols = pre.transformers_[0][2]
        feat_names = list(num_cols) + cat_features
    X_df = pd.DataFrame(X, columns=feat_names, index=base.index)

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{csv_path.stem}_{_timestamp()}"
    out_parquet = cfg.paths.processed_dir / f"{out_stem}.parquet"
    out_csv = cfg.paths.processed_dir / f"{out_stem}.csv"
    if tcfg.save_parquet: X_df.to_parquet(out_parquet, index=False)
    if tcfg.save_csv: X_df.to_csv(out_csv, index=False)

    rejects_path: Optional[Path] = None
    if not rejects.empty:
        rej_dir = (cfg.paths.interim_dir / "rejects"); rej_dir.mkdir(parents=True, exist_ok=True)
        rejects_path = rej_dir / f"{csv_path.stem}_{_timestamp()}_rejects.csv"
        rejects.to_csv(rejects_path, index=False)

    mark = {
        "source": csv_path.as_posix(),
        "processed_parquet": out_parquet.as_posix() if tcfg.save_parquet else None,
        "processed_csv": out_csv.as_posix() if tcfg.save_csv else None,
        "rejects": rejects_path.as_posix() if rejects_path else None,
        "report": rep.__dict__,
        "outlier_mode": tcfg.outlier_mode,
        "iqr_k": tcfg.iqr_k,
        "when": _timestamp(),
    }
    marker_path = csv_path.with_suffix(csv_path.suffix + ".done.json")
    marker_path.write_text(json.dumps(mark, indent=2), encoding="utf-8")
    save_json(mark, f"ingest_{csv_path.stem}.json", cfg, subdir="reports")
    logger.info("Ingested %s -> rows %d→%d | features %d | rejects %d",
                csv_path.name, rep.input_rows, rep.output_rows, X_df.shape[1], rep.rejects_rows)
    return mark


def scan_interim(
    cfg: Config,
    schema_path: str | Path = "config/schema.yaml",
    tcfg: TransformConfig = TransformConfig(),
) -> List[Dict[str, Any]]:
    logger = get_logger("ingest", cfg=cfg, to_file=True)
    schema = load_schema(schema_path)
    interim = cfg.paths.interim_dir; interim.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for csv_path in sorted(interim.glob("*.csv")):
        if csv_path.with_suffix(csv_path.suffix + ".done.json").exists(): continue
        try:
            results.append(process_one(csv_path, cfg, schema, logger, tcfg=tcfg))
        except Exception as e:
            logger.exception("Failed to process %s: %s", csv_path.name, e)
    if not results: logger.info("No pending CSVs found in %s.", interim)
    return results


# ----------------------------------- CLI -----------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Clean+transform interim CSVs → processed; rejects quarantined.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--schema", default="config/schema.yaml")
    ap.add_argument("--outliers", choices=["off", "iqr"], default="iqr")
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config); cfg.paths.ensure_dirs(include_figures=False)
        tcfg = TransformConfig(outlier_mode=args.outliers, iqr_k=args.iqr_k,
                               save_csv=bool(args.save_csv), save_parquet=not bool(args.no_parquet))
        out = scan_interim(cfg, schema_path=args.schema, tcfg=tcfg)
        print(json.dumps({"processed": out}, indent=2))
        return 0
    except Exception as e:
        print(f"[ingest] Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
