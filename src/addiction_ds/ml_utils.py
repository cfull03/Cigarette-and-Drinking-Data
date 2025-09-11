# File: src/addiction_ds/ml_utils.py
"""Minimal utilities for splitting a CSV into train/val.

Features
--------
- **Default**: splits the newest `*.csv` under `data/processed/`.
- **Override**: `--csv` lets you pick a specific file.
- Outputs default to `data/processed/train.csv` and `data/processed/val.csv`.
- Optional `--stratify-col` if you want a stratified split.

CLI
---
python -m addiction_ds.ml_utils split \
    [--csv path/to/data.csv] \
    [--dir data/processed] \
    [--out-train data/processed/train.csv] \
    [--out-val data/processed/val.csv] \
    [--test-size 0.2] [--random-state 42] [--stratify-col LABEL]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUT_TRAIN = DEFAULT_PROCESSED_DIR / "train.csv"
DEFAULT_OUT_VAL = DEFAULT_PROCESSED_DIR / "val.csv"


def _most_recent_csv(directory: Path) -> Path | None:
    """Return the most-recent CSV in *directory*, or None if none exist."""
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def split_csv(
    input_csv: Path,
    out_train: Path = DEFAULT_OUT_TRAIN,
    out_val: Path = DEFAULT_OUT_VAL,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> tuple[Path, Path]:
    """Split a single CSV into train/val CSV files without any preprocessing.

    Parameters
    ----------
    input_csv : Path
        The CSV to split.
    out_train, out_val : Path
        Output paths for the split CSVs.
    test_size : float
        Validation split size fraction (default 0.2).
    random_state : int
        Random seed for reproducibility (default 42).
    stratify_col : Optional[str]
        If provided and present in the CSV, stratify on this column.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv).drop_duplicates()

    strat = None
    if stratify_col and stratify_col in df.columns:
        strat = df[stratify_col]

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    return out_train, out_val


def _cli_split(args: argparse.Namespace) -> None:
    base_dir = Path(args.dir) if args.dir else DEFAULT_PROCESSED_DIR

    if args.csv:
        input_csv = Path(args.csv)
    else:
        input_csv = _most_recent_csv(base_dir)
        if input_csv is None:
            raise SystemExit(
                f"No CSVs found in {base_dir}. Provide --csv or place a file in that directory."
            )

    out_train = Path(args.out_train) if args.out_train else DEFAULT_OUT_TRAIN
    out_val = Path(args.out_val) if args.out_val else DEFAULT_OUT_VAL

    t, v = split_csv(
        input_csv=input_csv,
        out_train=out_train,
        out_val=out_val,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_col=args.stratify_col,
    )

    print(f"Wrote train → {t}\nWrote  val  → {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m addiction_ds.ml_utils",
        description=("Split a CSV into train/val (defaults to newest CSV in data/processed)."),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split", help="Split CSV into train/val")
    p_split.add_argument("--csv", default=None, help="Path to CSV to split (optional)")
    p_split.add_argument(
        "--dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory to search for the newest CSV when --csv is omitted",
    )
    p_split.add_argument(
        "--out-train", default=str(DEFAULT_OUT_TRAIN), help="Output train CSV path"
    )
    p_split.add_argument("--out-val", default=str(DEFAULT_OUT_VAL), help="Output val CSV path")
    p_split.add_argument("--test-size", type=float, default=0.2)
    p_split.add_argument("--random-state", type=int, default=42)
    p_split.add_argument(
        "--stratify-col",
        default=None,
        help="Optional column to stratify on if present in the CSV",
    )
    p_split.set_defaults(func=_cli_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
