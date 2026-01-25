# utils/targets/preprocessing.py
from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_release_cols(track_df: pd.DataFrame) -> pd.DataFrame:
    df = track_df.copy()
    for src, tgt in [("album_release_year", "release_year"), ("album_release_month", "release_month")]:
        if tgt not in df.columns and src in df.columns:
            df[tgt] = df[src]
    return df

def build_cohort_ym(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    cohort = (y * 100 + m).astype("Int64")
    return cohort

def ensure_release_month_ts(df: pd.DataFrame) -> pd.Series:
    """
    Robust: nimmt zuerst eine echte Datums-Spalte, sonst baut aus release_year/month.
    """
    date_cols = ["album_release_date_parsed", "release_date_parsed", "release_date"]
    used = next((c for c in date_cols if c in df.columns), None)

    if used:
        ts = pd.to_datetime(df[used], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return ts

    y = pd.to_numeric(df.get("release_year", np.nan), errors="coerce").astype("Int64")
    m = pd.to_numeric(df.get("release_month", np.nan), errors="coerce").astype("Int64")
    s = (y.astype("string") + "-" + m.astype("string") + "-01")
    return pd.to_datetime(s, errors="coerce")