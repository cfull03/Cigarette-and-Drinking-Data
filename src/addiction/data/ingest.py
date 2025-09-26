# file: src/addiction/data/ingest.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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
    return DatasetSchema(
        primary_key=list(data.get("primary_key", [])),
        fields=fields,
    )


# ----------------------------- cleaning & coercion -----------------------------

def _coerce_column(s: pd.Series, rule: FieldRule) -> pd.Series:
    if rule.type == "integer":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if rule.type == "number":
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if rule.type == "boolean":
        true = {"true","1","yes","y","t"}
        false = {"false","0","no","n","f"}
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

    # Align columns to schema
    want_cols = list(schema.fields.keys())
    for c in want_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[want_cols].copy()

    # Coerce types
    for name, rule in schema.fields.items():
        df[name] = _coerce_column(df[name], rule)

    # Valid mask + reason collector
    valid = pd.Series(True, index=df.index)
    reasons: List[List[str]] = [[] for _ in range(len(df))]

    dropped_nullability = 0
    dropped_enum = 0
    dropped_range = 0
    dropped_pk_dupes = 0

    # Nullability
    for name, rule in schema.fields.items():
        if not rule.nullable:
            mask_bad = df[name].isna()
            if mask_bad.any():
                dropped_nullability += int(mask_bad.sum())
                valid &= ~mask_bad
                idxs = df.index[mask_bad]
                for i in idxs:
                    reasons[i].append(f"nullability:{name}")

    # Enums
    for name, rule in schema.fields.items():
        if rule.enum is not None:
            allowed = set(str(v) for v in rule.enum)
            mask_bad = (~df[name].isna()) & (~df[name].astype("string").isin(allowed))
            if mask_bad.any():
                dropped_enum += int(mask_bad.sum())
                valid &= ~mask_bad
                for i in df.index[mask_bad]:
                    reasons[i].append(f"enum:{name}")

    # Numeric ranges
    for name, rule in schema.fields.items():
        if rule.type in {"integer", "number"}:
            mask_bad = pd.Series(False, index=df.index)
            if rule.min is not None:
                mask_bad |= (~df[name].isna()) & (df[name] < rule.min)
            if rule.max is not None:
                mask_bad |= (~df[name].isna()) & (df[name] > rule.max)
            if mask_bad.any():
                dropped_range += int(mask_bad.sum())
                valid &= ~mask_bad
                for i in df.index[mask_bad]:
                    reasons[i].append(f"range:{name}")

    # Split valid vs rejects before PK dedupe to preserve all failing rows
    cleaned = df[valid].copy()
    rejects = df[~valid].copy()

    # Primary key dedupe on cleaned
    if schema.primary_key and not cleaned.empty:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=schema.primary_key, keep="first")
        dropped_pk_dupes = before - len(cleaned)
        if dropped_pk_dupes > 0:
            # Mark duplicates as rejects with reason
            dup_mask = ~cleaned.index.isin(df[valid].drop_duplicates(subset=schema.primary_key, keep="first").index)
            # Above trick already removed dups; to record them, recompute on original 'valid' rows:
            valid_df = df[valid]
            dup_rows = valid_df[valid_df.duplicated(subset=schema.primary_key, keep="first")]
            if not dup_rows.empty:
                dup_rows = dup_rows.copy()
                dup_rows["reject_reasons"] = "pk_duplicate"
                rejects = pd.concat([rejects, dup_rows], axis=0, ignore_index=True)

    # Attach reasons to rejects
    if not rejects.empty and "reject_reasons" not in rejects.columns:
        rr = []
        for i in rejects.index:
            # Original index might not align after filtering; use dict lookup
            orig_i = rejects.index[i]
            rr.append(";".join(reasons[orig_i]) if orig_i < len(reasons) else "unknown")
        rejects = rejects.copy()
        rejects["reject_reasons"] = rr

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


# ----------------------------- ingest scanning -----------------------------

def _timestamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def process_one(csv_path: Path, cfg: Config, schema: DatasetSchema, logger) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    cleaned, rejects, rep = clean_dataframe(df, schema)

    # Save cleaned -> processed (timestamped)
    out_name = f"{csv_path.stem}_{_timestamp()}.csv"
    out_path = (cfg.paths.processed_dir / out_name).resolve()
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    # Save rejects -> interim/rejects (timestamped) if any
    rejects_path: Optional[Path] = None
    if not rejects.empty:
        rej_dir = (cfg.paths.interim_dir / "rejects").resolve()
        rej_dir.mkdir(parents=True, exist_ok=True)
        rejects_name = f"{csv_path.stem}_{_timestamp()}_rejects.csv"
        rejects_path = rej_dir / rejects_name
        rejects.to_csv(rejects_path, index=False)

    # Marker JSON next to interim file
    mark = {
        "source": csv_path.as_posix(),
        "output_clean": out_path.as_posix(),
        "output_rejects": rejects_path.as_posix() if rejects_path else None,
        "report": rep.__dict__,
        "when": _timestamp(),
    }
    marker_path = csv_path.with_suffix(csv_path.suffix + ".done.json")
    marker_path.write_text(json.dumps(mark, indent=2), encoding="utf-8")

    # Log a summary into reports
    save_json(mark, f"ingest_{csv_path.stem}.json", cfg, subdir="reports")
    logger.info(
        "Ingested %s -> %s (kept %d/%d, rejects %d)",
        csv_path.name, out_path.name, rep.output_rows, rep.input_rows, rep.rejects_rows
    )
    return mark


def scan_interim(cfg: Config, schema_path: str | Path = "config/schema.yaml") -> List[Dict[str, Any]]:
    logger = get_logger("ingest", cfg=cfg, to_file=True)
    schema = load_schema(schema_path)
    interim = cfg.paths.interim_dir
    interim.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for csv_path in sorted(interim.glob("*.csv")):
        # skip already marked
        if csv_path.with_suffix(csv_path.suffix + ".done.json").exists():
            continue
        try:
            results.append(process_one(csv_path, cfg, schema, logger))
        except Exception as e:
            logger.exception("Failed to process %s: %s", csv_path.name, e)
    if not results:
        logger.info("No pending CSVs found in %s.", interim)
    return results


# ----------------------------------- CLI -----------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Validate & clean interim CSVs and move them to processed; save rejects to quarantine.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--schema", default="config/schema.yaml")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        cfg.paths.ensure_dirs(include_figures=False)
        out = scan_interim(cfg, schema_path=args.schema)
        print(json.dumps({"processed": out}, indent=2))
        return 0
    except Exception as e:
        print(f"[ingest] Failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
