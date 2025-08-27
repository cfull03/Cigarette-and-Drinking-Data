from __future__ import annotations
import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: normalize headers; drop all-NA columns. Safe by default."""
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("&", "and", regex=False)
        )
    out = out.dropna(axis=1, how="all")
    return out