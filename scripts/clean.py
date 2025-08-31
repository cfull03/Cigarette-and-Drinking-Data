#!/usr/bin/env python3
"""
Config-driven cleaner with CI-friendly multi-file support and text sanitization.
- Works on --input/--glob, or falls back to cfg.paths.raw -> cfg.paths.sample_input.
- Sanitizes ALL string-like columns (unicode NFKC, strip, collapse whitespace, strip quotes, remove control/zero-width).
- Applies optional rules under `cleaning:` in configs/default.yaml.
- Writes to interim & processed via project IO helpers; falls back to direct CSV writes if helpers are missing.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

__all__ = ["main"]


# ---------------- CLI ----------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean dataset(s) and write outputs")
    p.add_argument("--module", default=os.environ.get("MODULE", "addiction_ds"))
    p.add_argument("--config", default=os.environ.get("CONFIG", "configs/default.yaml"))
    p.add_argument("--stem", default=os.environ.get("STEM", "auto"), help="'auto' derives from each input filename")
    p.add_argument("--input", "-i", action="append", default=[], help="CSV input path (repeatable)")
    p.add_argument("--glob", action="append", default=[], help="Glob(s), e.g. 'data/raw/*.csv'")
    p.add_argument("--index-col")
    p.add_argument("--income-col")
    p.add_argument("--age-col")
    p.add_argument("--sleep-hours-col")
    p.add_argument("--bmi-col")
    p.add_argument("--round-income-to", type=int)
    p.add_argument("--no-qcut", action="store_true")
    return p.parse_args(argv)


# ------------- utils -------------

def _mode_or_nan(s: Iterable[Any]) -> Any:
    try:
        m = s.mode()
        return m.iloc[0] if len(m) else None
    except Exception:
        return None


def _ensure_list(x: Any | None) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _gather_inputs(args) -> List[Path]:
    files: List[Path] = [Path(p) for p in args.input]
    for pat in args.glob:
        for p in Path().glob(pat):
            if p.suffix.lower() == ".csv":
                files.append(p)
    if not files:
        return []
    seen: set[Path] = set()
    out: List[Path] = []
    for p in files:
        if p.exists() and p not in seen:
            out.append(p)
            seen.add(p)
    return out


# ------ text sanitization ------

def _sanitize_text_series(s, *, pd, norm: str | None, collapse_ws: bool, strip_quotes: bool, remove_ctrl: bool):
    s = s.astype("string")
    if norm:
        try:
            s = s.str.normalize(norm)
        except Exception:
            pass
    if remove_ctrl:
        s = s.str.replace(r"[\u0000-\u001F\u007F-\u009F\u200B-\u200D\uFEFF]", "", regex=True)
    if collapse_ws:
        s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    if strip_quotes:
        s = s.str.replace(r"^['\"](.*)['\"]$", r"\1", regex=True)
    return s


def _sanitize_text_columns(df, C, *, pd):
    T = C.get("text", {})
    if T.get("enable", True) is False:
        return df
    include = _ensure_list(T.get("columns"))
    exclude = set(_ensure_list(T.get("exclude")))
    norm = T.get("normalize_unicode", "NFKC")
    collapse_ws = T.get("collapse_whitespace", True)
    strip_quotes = T.get("strip_quotes", True)
    remove_ctrl = T.get("remove_control_chars", True)

    import pandas as pd_  # dtype checks

    targets = [c for c in df.columns if (pd_.api.types.is_string_dtype(df[c]) or df[c].dtype == "object")]
    if include:
        targets = [c for c in include if c in df.columns]

    for c in targets:
        if c in exclude:
            continue
        ser = df[c]
        if str(ser.dtype) == "category":
            ser = ser.astype("string")
            df[c] = _sanitize_text_series(ser, pd=pd, norm=norm, collapse_ws=collapse_ws, strip_quotes=strip_quotes, remove_ctrl=remove_ctrl).astype("category")
        else:
            df[c] = _sanitize_text_series(ser, pd=pd, norm=norm, collapse_ws=collapse_ws, strip_quotes=strip_quotes, remove_ctrl=remove_ctrl)
    return df


# ------------- cleaning -------------

def _clean(df, cfg: Dict[str, Any], *, np, pd):
    C = cfg.get("cleaning", {})
    out = df.copy()

    if C.get("normalize_columns", True):
        out.columns = (
            out.columns.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("&", "and", regex=False)
        )

    out = _sanitize_text_columns(out, C, pd=pd)

    for col in _ensure_list(C.get("categories")):
        if col in out.columns:
            out[col] = out[col].astype("category")

    income = C.get("income", {})
    income_col = income.get("col")
    if income_col and income_col in out.columns:
        out[income_col] = pd.to_numeric(out[income_col], errors="coerce")
        round_to = income.get("round_to")
        if isinstance(round_to, int) and round_to > 0:
            out[income_col] = (np.round(out[income_col] / round_to) * round_to).astype("Float64")
        q = income.get("qcut")
        labels = income.get("labels")
        feature = income.get("feature", "income_percentile")
        if q and labels and len(labels) == len(q) - 1:
            try:
                out[feature] = pd.qcut(out[income_col], q=q, labels=labels, duplicates="drop")
                out[feature] = out[feature].astype("category")
            except Exception:
                pass

    age = C.get("age", {})
    age_col = age.get("col")
    if age_col and age_col in out.columns:
        out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
        bins = age.get("bins", [])
        labels = age.get("labels", [])
        feature = age.get("feature", "age_group")
        if bins and labels and len(labels) == len(bins) - 1:
            _bins = [(-math.inf if str(b).lower() in {"-inf", "-infinity"} else (math.inf if str(b).lower() in {"inf", "infinity"} else float(b))) for b in bins]
            try:
                out[feature] = pd.cut(out[age_col], bins=_bins, labels=labels, include_lowest=True)
                out[feature] = out[feature].astype("category")
            except Exception:
                pass

    sleep = C.get("sleep", {})
    hours_col = sleep.get("hours_col")
    threshold = sleep.get("threshold", 8)
    feature = sleep.get("feature", "adequate_sleep")
    if hours_col and hours_col in out.columns:
        try:
            out[feature] = (
                np.where(pd.to_numeric(out[hours_col], errors="coerce") > threshold, "Adequate Sleep", "Not Adequate Sleep")
                .astype("string").astype("category")
            )
        except Exception:
            pass

    for rule in _ensure_list(C.get("imputations")):
        target = rule.get("target")
        by = [c for c in _ensure_list(rule.get("by")) if c in out.columns]
        method = (rule.get("method") or "mode").lower()
        if not target or target not in out.columns or not by:
            continue
        g = out.groupby(by, observed=True)
        if method == "mode":
            out[target] = out[target].fillna(g[target].transform(_mode_or_nan))
        elif method in {"median", "mean"}:
            func = (lambda s: s.median()) if method == "median" else (lambda s: s.mean())
            out[target] = pd.to_numeric(out[target], errors="coerce")
            out[target] = out[target].fillna(g[target].transform(func))
        elif method == "constant":
            out[target] = out[target].fillna(rule.get("value"))

    index_col = C.get("index_col")
    if index_col and index_col in out.columns:
        out = out.set_index(index_col)

    dedup = _ensure_list(C.get("drop_duplicates_by"))
    if dedup and all(c in out.columns or c == out.index.name for c in dedup):
        out = out.drop_duplicates(subset=dedup)

    keep = C.get("keep_columns")
    drop = _ensure_list(C.get("drop_columns"))
    if keep:
        keep_list = [c for c in _ensure_list(keep) if (c in out.columns or c == out.index.name)]
        if out.index.name and out.index.name not in keep_list:
            out = out.reset_index()
        out = out[keep_list]
    elif drop:
        drop_list = [c for c in drop if c in out.columns]
        if drop_list:
            out = out.drop(columns=drop_list)

    return out


# -------- driver --------

def _ensure_dir_compat(io_mod, p) -> None:
    fn = getattr(io_mod, "ensure_dir", None)
    if fn is not None:
        try:
            fn(p)
            return
        except Exception:
            pass
    Path(p).mkdir(parents=True, exist_ok=True)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        io = importlib.import_module(f"{args.module}.io")
    except Exception as exc:
        print(f"Failed to import '{args.module}.io': {exc}", file=sys.stderr)
        return 2

    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    cfg: Dict[str, Any] = io.load_cfg(args.config)

    C = cfg.setdefault("cleaning", {})
    if args.index_col:
        C["index_col"] = args.index_col
    if args.income_col or args.round_income_to is not None or args.no_qcut:
        inc = C.setdefault("income", {})
        if args.income_col:
            inc["col"] = args.income_col
        if args.round_income_to is not None:
            inc["round_to"] = args.round_income_to
        if args.no_qcut:
            inc["qcut"] = None; inc["labels"] = None
    if args.age_col:
        C.setdefault("age", {})["col"] = args.age_col
    if args.sleep_hours_col:
        C.setdefault("sleep", {})["hours_col"] = args.sleep_hours_col
    if args.bmi_col:
        C.setdefault("sleep", {})["bmi_col"] = args.bmi_col

    P: Dict[str, Any] = io.get_paths(cfg)

    inputs = _gather_inputs(args)
    if not inputs:
        src = P.get("raw")
        if not getattr(src, "exists", lambda: False)():
            src = P.get("sample_input")
        if src is None or not getattr(src, "exists", lambda: False)():
            print("No valid input file found (raw or sample_input), and no --input/--glob provided", file=sys.stderr)
            return 2
        inputs = [Path(src)]

    _ensure_dir_compat(io, P["interim_dir"])  # no dependency on project helper
    _ensure_dir_compat(io, P["processed_dir"])  # no dependency on project helper

    ok = True
    for inp in inputs:
        print(f"Loading: {inp}")
        try:
            df = io.read_csv(inp)
        except Exception as exc:
            print(f"Failed reading {inp}: {exc}", file=sys.stderr)
            ok = False
            continue

        df_clean = _clean(df, cfg, np=np, pd=pd)

        stem = args.stem
        if stem == "auto" or not stem:
            stem = f"{inp.stem}_clean"

        # interim
        try:
            interim_path = io.to_interim(df_clean, cfg, stem=stem, index=True)
        except Exception:
            interim_path = Path(P["interim_dir"]) / f"{stem}.csv"
            df_clean.to_csv(interim_path, index=True)
        print("interim ->", interim_path)

        # processed
        try:
            processed_path = io.to_processed(df_clean, cfg, stem=f"{stem}_features_v1", index=True)
        except Exception:
            processed_path = Path(P["processed_dir"]) / f"{stem}_features_v1.csv"
            df_clean.to_csv(processed_path, index=True)
        print("processed ->", processed_path)

        # sample
        try:
            sample_path = io.write_sample(df_clean, cfg, n=100, index=False)
            print("sample ->", sample_path)
        except Exception:
            pass

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
