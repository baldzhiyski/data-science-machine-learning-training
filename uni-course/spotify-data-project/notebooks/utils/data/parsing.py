import ast
import json
from typing import Optional

import numpy as np
import pandas as pd
#%%
def parse_datetime_from_candidates(df: pd.DataFrame, candidates: list[str], out_col: str) -> pd.DataFrame:
    """
    Pick the first existing column from `candidates` and parse it to datetime.
    If the chosen column is already datetime-like, keep it as-is.
    If none exists, create out_col as NaT.
    """
    src = None
    for c in candidates:
        if c in df.columns:
            src = c
            break

    if src is None:
        df[out_col] = pd.NaT
        return df

    s = df[src]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[out_col] = s
    else:
        df[out_col] = pd.to_datetime(s.astype("string"), errors="coerce")

    return df

def col_or_na(df: pd.DataFrame, col: str, dtype: Optional[str] = None) -> pd.Series:
    """
    Zweck:
    - Sichere Hilfsfunktion, um eine Spalte aus einem DataFrame zu holen.
    - Falls die Spalte nicht existiert, wird stattdessen eine Series mit NA-Werten (gleiche Länge/Index) zurückgegeben.
    Warum?
    - In realen Datensätzen fehlen manchmal Spalten (z.B. bei Batch-Scoring).
    - So verhindert man KeyErrors und die Pipeline bleibt robust.

    Parameter:
    - df: pandas DataFrame
    - col: Name der gewünschten Spalte
    - dtype: optionaler Ziel-Datentyp (z.B. "float64", "string")

    Rückgabe:
    - Immer eine pandas Series (nie None).
    """
    # Sicherheitscheck: df muss wirklich ein DataFrame sein, sonst ist die Nutzung fehlerhaft.
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("col_or_na: df must be a pandas DataFrame")

    # Falls die Spalte existiert: hole sie.
    if col in df.columns:
        s = df[col]

        # Optional: versuche die Series auf den gewünschten dtype zu casten.
        # Wenn das Casting scheitert, ignorieren wir den Fehler (Pipeline soll nicht abbrechen).
        if dtype is not None:
            try:
                s = s.astype(dtype)
            except Exception:
                pass

        return s

    # Falls die Spalte fehlt: gib eine NA-Serie zurück, aber mit gleichem Index wie df.
    # Vorteil: Downstream-Code kann weiterhin damit rechnen, dass die Länge passt.
    return pd.Series(pd.NA, index=df.index)

def ensure_list_column(s: pd.Series) -> pd.Series:
    """
    Zweck:
    - Stellt sicher, dass jede Zelle in der Spalte eine echte Python-Liste ist.
    - Sehr wichtig für Features wie Genres/Tags, die manchmal als Strings gespeichert sind.

    Akzeptierte Eingaben pro Zelle:
    - bereits echte Listen: ["pop", "rock"]
    - JSON-String: '["pop","rock"]'
    - Python-repr-String: "['pop', 'rock']"
    - None/NaN: wird zu []

    Rückgabe:
    - Series, in der jede Zeile garantiert eine Liste (oder []) enthält.
    """

    def parse_one(v):
        # Fall 1: schon eine Liste -> direkt zurück.
        if isinstance(v, list):
            return v

        # Fall 2: None oder NaN -> leere Liste.
        # np.isnan funktioniert nur für floats; daher prüfen wir vorher den Typ.
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []

        # Fall 3: String -> mehrere Parser-Versuche.
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []

            # Versuch A: JSON parsen (sicherer als eval).
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            # Versuch B: Python-Literal parsen (z.B. "['a','b']").
            # ast.literal_eval ist viel sicherer als eval (führt keinen Code aus).
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        # Wenn alles fehlschlägt: leere Liste als Default (robust für Pipelines).
        return []

    return s.apply(parse_one)

def parse_release_date_universal(s: pd.Series) -> pd.Series:
    """
    Universal release_date parser that parses EVERYTHING it reasonably can.

    Supports:
      - Spotify date strings: "YYYY", "YYYY-MM", "YYYY-MM-DD"
      - Epoch timestamps (string or numeric), incl. negative and old:
          * >= 11 digits  -> milliseconds
          * 9-10 digits   -> seconds
      - 0 treated as missing (NaT) to avoid fake 1970-01-01

    Returns:
      datetime64[ns] Series with NaT for unparseable values.
    """
    x = s.astype("string").str.strip()
    x = x.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})

    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")

    # ---------- numeric epoch parsing ----------
    num = pd.to_numeric(x, errors="coerce")

    # treat 0 as missing (placeholder -> avoids 1970-01-01 pollution)
    num = num.mask(num == 0)

    # integer coercion
    num_int = pd.Series(pd.NA, index=x.index, dtype="Int64")
    mask_num = num.notna()
    if mask_num.any():
        num_int.loc[mask_num] = np.floor(num.loc[mask_num].astype("float64")).astype("int64")

    # digit-length heuristic (works for old/negative ms values too)
    digits = num_int.abs().astype("string").str.len()
    ms_mask = num_int.notna() & (digits >= 11)          # milliseconds
    s_mask  = num_int.notna() & digits.between(9, 10)   # seconds

    out.loc[ms_mask] = pd.to_datetime(num_int.loc[ms_mask].astype("int64"), unit="ms", errors="coerce")
    out.loc[s_mask]  = pd.to_datetime(num_int.loc[s_mask].astype("int64"), unit="s", errors="coerce")

    # ---------- spotify-like string parsing ----------
    rest = out.isna() & x.notna()

    txt = x.copy()
    txt = txt.where(~txt.str.fullmatch(r"\d{4}"), txt + "-01-01")
    txt = txt.where(~txt.str.fullmatch(r"\d{4}-\d{2}"), txt + "-01")

    out.loc[rest] = pd.to_datetime(txt.loc[rest], errors="coerce")

    return out