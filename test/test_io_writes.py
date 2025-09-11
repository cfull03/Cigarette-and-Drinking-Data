# File: tests/test_io_writes.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from addiction_ds.io import to_interim, to_processed, write_sample


def test_io_saves_versioned_csvs(tmp_path: Path):
    cfg = {
        "paths": {
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "sample_input": str(tmp_path / "sample" / "sample_100_rows.csv"),
        }
    }

    df = pd.DataFrame({"x": [1, 2, 3]})

    p1 = Path(to_interim(df, cfg, stem="clean"))
    p2 = Path(to_processed(df, cfg, stem="features_v1"))
    p3 = Path(write_sample(df, cfg, n=2))

    assert p1.exists() and p1.suffix == ".csv"
    assert p2.exists() and p2.suffix == ".csv"
    assert p3.exists() and p3.suffix == ".csv"
