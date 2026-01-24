from typing import Tuple

import pandas as pd
from ..data.parsing import col_or_na


def pick_release_cols_fast(df: pd.DataFrame) -> Tuple[str, str]:
    if {"release_year", "release_month"}.issubset(df.columns):
        return "release_year", "release_month"
    raise ValueError("Need release_year + release_month columns.")

def add_release_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Zweck:
    - Erzeugt aus einer Datums-Spalte zusätzliche Zeit-Features:
      * release_year   (Jahr)
      * release_month  (Monat)
      * release_decade (Dekade, z.B. 1990, 2000, 2010)

    Robustheit:
    - Unterstützt echte Datumsstrings (YYYY / YYYY-MM / YYYY-MM-DD)
    - Unterstützt Unix-Epoch als Zahl (Sekunden oder Millisekunden), auch NEGATIV (vor 1970),
      z.B. -473385600000

    Parameter:
    - df: DataFrame
    - date_col: Name der Spalte mit Datum/Datetime-ähnlichen Werten

    Rückgabe:
    - Kopie von df, erweitert um neue Spalten (Original bleibt unverändert).
    """
    df = df.copy()

    s = col_or_na(df, date_col)

    # --- 1) Versuche numeric epoch parsing (ms oder s) ---
    num = pd.to_numeric(s, errors="coerce")
    dt_num = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    is_num = num.notna()
    # Heuristik: ms ~ 1e12, s ~ 1e9 (heute); alles >= 1e10 behandeln wir als ms
    is_ms = is_num & (num.abs() >= 1e10)
    is_s  = is_num & ~is_ms

    if is_ms.any():
        dt_num.loc[is_ms] = pd.to_datetime(num.loc[is_ms], unit="ms", origin="unix", errors="coerce")
    if is_s.any():
        dt_num.loc[is_s] = pd.to_datetime(num.loc[is_s], unit="s", origin="unix", errors="coerce")

    # --- 2) Fallback: String parsing (ISO-like) ---
    dt_str = pd.to_datetime(s.astype("string"), errors="coerce")

    # Combine: prefer numeric epoch parse, fallback to string parse
    dt = dt_num.combine_first(dt_str)

    # Features (nullable Int64)
    df["release_year"] = dt.dt.year.astype("Int64")
    df["release_month"] = dt.dt.month.astype("Int64")
    df["release_decade"] = ((dt.dt.year // 10) * 10).astype("Int64")

    return df

