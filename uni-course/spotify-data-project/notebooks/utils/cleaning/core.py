# utils/cleaning/core.py
"""
Kernmodul für Data Cleaning: Policy + Hilfsfunktionen + Profiling + Cleaner-Klassen.

Ziele:
- Reproduzierbares, regelbasiertes Cleaning (Data Contract)
- Notebook-freundlich (keine Abhängigkeit von Notebook-Globals)
- Einheitliche, nachvollziehbare Transformationen mit klaren Regeln
"""

from __future__ import annotations

import math
import re
import time
import platform
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd


# =============================================================================
# Policy / Konfiguration
# =============================================================================

@dataclass(frozen=True)
class CleaningPolicy:
    """
    Zentrale Policy (Datenvertrag) für das Cleaning.

    Enthält Schalter und Grenzwerte für Plausibilitätschecks, Clipping
    und Ausreißer-Behandlung.
    """
    drop_orphan_bridge_rows: bool = True

    # Popularity-Validierung (Spotify-typisch 0..100)
    clip_popularity: bool = True
    popularity_min: int = 0
    popularity_max: int = 100

    # Quantil-Caps für extreme Ausreißer (z.B. duration, tempo)
    duration_cap_quantile: float = 0.999
    tempo_cap_quantile: float = 0.999

    # Audio-Features im [0,1]-Bereich
    audio_01_cols: Tuple[str, ...] = (
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "speechiness", "valence"
    )

    # Weitere typische Feature-Ranges
    loudness_range: Tuple[float, float] = (-60.0, 5.0)
    key_range: Tuple[int, int] = (0, 11)
    mode_range: Tuple[int, int] = (0, 1)


# Default-Policy (kann im Notebook überschrieben werden)
POLICY = CleaningPolicy()


def build_run_meta(paths: Any, policy: CleaningPolicy, random_seed: int) -> Dict[str, Any]:
    """
    Erzeugt Laufzeit-Metadaten für Reports/Tracking.

    Hinweis:
    - `paths` ist häufig ein Dataclass-Objekt (asdict-fähig) oder ein dict.
    - Durch die Funktion entsteht keine Abhängigkeit von Notebook-Globals beim Import.
    """
    return {
        "run_ts_unix": int(time.time()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "random_seed": int(random_seed),
        "paths": {k: str(v) for k, v in asdict(paths).items()} if hasattr(paths, "__dataclass_fields__") else dict(paths),
        "policy": asdict(policy),
    }


# =============================================================================
# String / Typ Utilities
# =============================================================================

def snake_case(s: str) -> str:
    """Konvertiert einen String robust in snake_case (z.B. für Spaltennamen)."""
    s = s.strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_").lower()


def norm_str(s: pd.Series) -> pd.Series:
    """
    Normalisiert Whitespace und mappt leere Strings auf NA.

    Beispiel:
    "  Foo   Bar  " -> "Foo Bar"
    ""              -> <NA>
    """
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace("", pd.NA)
    return s


def to_int(s: pd.Series) -> pd.Series:
    """Robuste Integer-Konvertierung (pandas Int64 unterstützt NA)."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def to_float(s: pd.Series) -> pd.Series:
    """Robuste Float-Konvertierung."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def to_bool(s: pd.Series) -> pd.Series:
    """
    Robuster Boolean-Parser auf pandas BooleanDtype.

    Akzeptiert u.a.:
    True:  1, true, t, yes, y
    False: 0, false, f, no, n
    """
    x = s.astype("string").str.lower().str.strip()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[x.isin(["1", "true", "t", "yes", "y"])] = True
    out[x.isin(["0", "false", "f", "no", "n"])] = False
    return out


# =============================================================================
# Allgemeine Helpers
# =============================================================================

def memory_mb(df: pd.DataFrame) -> float:
    """Speicherbedarf eines DataFrames in MB (deep=True berücksichtigt Strings)."""
    return float(df.memory_usage(deep=True).sum()) / (1024 ** 2)


def clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clipped eine Series auf [lo, hi] (NA bleibt NA)."""
    return s.clip(lower=lo, upper=hi)


def safe_quantile_cap(s: pd.Series, q: float) -> Optional[float]:
    """
    Liefert einen robusten Quantil-Cap oder None.

    Warum?
    - quantile() kann bei leeren Serien/NA-only None/NaN liefern
    - wir wollen kein 'if cap' (weil cap=0 sonst fälschlich False wäre)
    """
    if s is None:
        return None
    x = pd.to_numeric(s, errors="coerce")
    x = x.dropna()
    if x.empty:
        return None
    cap = float(x.quantile(q))
    if math.isnan(cap):
        return None
    return cap


def keep_most_complete_row(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Entfernt Duplikate anhand key_cols und behält die "vollständigste" Zeile
    (höchste Anzahl nicht-null Werte).

    Nützlich, wenn Export/Join doppelte Records erzeugt, aber eine Zeile
    mehr Informationen enthält als die andere.
    """
    df = df.copy()
    df["_nonnulls"] = df.notna().sum(axis=1)
    df = df.sort_values("_nonnulls", ascending=False)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    df = df.drop(columns=["_nonnulls"])
    return df.reset_index(drop=True)


# =============================================================================
# Release Date Parser
# =============================================================================

def parse_release_date_universal(s: pd.Series) -> pd.Series:
    """
    Universeller Release-Date Parser: parsed "alles", was sinnvoll geht.

    Unterstützt:
      - Spotify-Format: "YYYY", "YYYY-MM", "YYYY-MM-DD"
      - Epoch Timestamps (als String oder Zahl), inkl. negative/alte:
          * >= 11 Ziffern  -> Millisekunden
          * 9-10 Ziffern   -> Sekunden
      - 0 wird als missing (NaT) behandelt (verhindert Fake 1970-01-01)

    Rückgabe:
      pandas Series datetime64[ns] mit NaT für nicht parsbare Werte.
    """
    x = s.astype("string").str.strip()
    x = x.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})

    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")

    # ---------- numerisches Epoch Parsing ----------
    num = pd.to_numeric(x, errors="coerce")

    # 0 als Platzhalter -> missing (sonst 1970-01-01)
    num = num.mask(num == 0)

    # Integer-Cast (über float, um z.B. "1234.0" zu handlen)
    num_int = pd.Series(pd.NA, index=x.index, dtype="Int64")
    mask_num = num.notna()
    if mask_num.any():
        num_int.loc[mask_num] = np.floor(num.loc[mask_num].astype("float64")).astype("int64")

    # Heuristik über Ziffernlänge (funktioniert auch für negative ms)
    digits = num_int.abs().astype("string").str.len()
    ms_mask = num_int.notna() & (digits >= 11)          # Millisekunden
    s_mask  = num_int.notna() & digits.between(9, 10)   # Sekunden

    out.loc[ms_mask] = pd.to_datetime(num_int.loc[ms_mask].astype("int64"), unit="ms", errors="coerce")
    out.loc[s_mask]  = pd.to_datetime(num_int.loc[s_mask].astype("int64"), unit="s", errors="coerce")

    # ---------- Spotify-ähnliche String-Formate ----------
    rest = out.isna() & x.notna()

    txt = x.copy()
    # "YYYY" -> "YYYY-01-01"
    txt = txt.where(~txt.str.fullmatch(r"\d{4}"), txt + "-01-01")
    # "YYYY-MM" -> "YYYY-MM-01"
    txt = txt.where(~txt.str.fullmatch(r"\d{4}-\d{2}"), txt + "-01")

    out.loc[rest] = pd.to_datetime(txt.loc[rest], errors="coerce",format="%Y-%m-%d")
    return out


# =============================================================================
# Quality Gates / Assertions
# =============================================================================

def assert_gate(condition: bool, msg: str):
    """
    Quality Gate: bricht den Lauf hart ab, wenn eine Kernannahme verletzt ist.
    """
    if not condition:
        raise AssertionError(f"QUALITY GATE FAILED: {msg}")


# =============================================================================
# Profiling (für Reports / Monitoring)
# =============================================================================

@dataclass
class TableProfile:
    """Struktur für kompakte Table-Statistiken (Report/Logging)."""
    name: str
    rows: int
    cols: int
    memory_mb: float
    missing_by_col: Dict[str, int]
    duplicate_rows_full: Optional[int] = None
    duplicate_rows_on_keys: Optional[int] = None


def profile_table(df: pd.DataFrame, name: str, key_cols: Optional[List[str]] = None) -> TableProfile:
    """
    Erstellt ein kompaktes Profil einer Tabelle:
    - Zeilen/Spalten
    - Speicher
    - Missing Values je Spalte
    - Duplikate (vollständig oder auf Keys)
    """
    missing = {c: int(df[c].isna().sum()) for c in df.columns}

    prof = TableProfile(
        name=name,
        rows=int(len(df)),
        cols=int(df.shape[1]),
        memory_mb=round(memory_mb(df), 2),
        missing_by_col=missing,
    )

    if key_cols:
        prof.duplicate_rows_on_keys = int(df.duplicated(subset=key_cols).sum())
    else:
        prof.duplicate_rows_full = int(df.duplicated().sum())

    return prof


# =============================================================================
# Cleaner-Klassen (tabellenspezifische Data-Contract-Implementierung)
# =============================================================================

class BaseCleaner:
    """
    Basisklasse für table-spezifische Cleaner.

    Erwartung:
    - `name` entspricht dem Tabellennamen (Key im raw-Dict).
    - `clean(df)` liefert einen bereinigten DataFrame zurück.
    """
    name: str

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TracksCleaner(BaseCleaner):
    """Cleaning-Regeln für Tabelle `tracks`."""
    name = "tracks"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure PK column name: track_id (Exporter kann "id" liefern)
        if "track_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "track_id"})

        # Pflichtfeld: track_id
        if "track_id" not in df.columns:
            # Harte Fehlermeldung: ohne PK ist die Tabelle wertlos
            raise ValueError("tracks: Missing primary key column 'track_id' (or 'id').")

        # Strings normalisieren
        df["track_id"] = norm_str(df["track_id"])
        df = df[df["track_id"].notna()]

        for c in ["name", "preview_url", "audio_feature_id"]:
            if c in df.columns:
                df[c] = norm_str(df[c])

        # Numerische Felder
        for c in ["disc_number", "track_number", "duration", "popularity"]:
            if c in df.columns:
                df[c] = to_int(df[c])

        # Booleans
        if "explicit" in df.columns:
            df["explicit"] = to_bool(df["explicit"])
        if "is_playable" in df.columns:
            df["is_playable"] = to_bool(df["is_playable"])

        # Regel: duration > 0, cap extremes (Quantil)
        if "duration" in df.columns:
            df.loc[df["duration"] <= 0, "duration"] = pd.NA
            cap = safe_quantile_cap(df["duration"], POLICY.duration_cap_quantile)
            if cap is not None:
                df["duration"] = df["duration"].clip(upper=int(cap))

        # Regel: popularity in [0, 100]
        if "popularity" in df.columns and POLICY.clip_popularity:
            df["popularity"] = clip_series(df["popularity"], POLICY.popularity_min, POLICY.popularity_max)

        # Duplikate auf track_id: behalte "vollständigste" Zeile
        df = keep_most_complete_row(df, ["track_id"])
        return df


class AudioFeaturesCleaner(BaseCleaner):
    """Cleaning-Regeln für Tabelle `audio_features`."""
    name = "audio_features"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Pflichtfeld: id
        if "id" not in df.columns:
            raise ValueError("audio_features: Missing primary key column 'id'.")

        df["id"] = norm_str(df["id"])
        df = df[df["id"].notna()]

        # Text/URL-Spalten
        for c in ["analysis_url"]:
            if c in df.columns:
                df[c] = norm_str(df[c])

        # floats [0,1]
        for c in POLICY.audio_01_cols:
            if c in df.columns:
                df[c] = clip_series(to_float(df[c]), 0.0, 1.0)

        # tempo: >0 + cap
        if "tempo" in df.columns:
            df["tempo"] = to_float(df["tempo"])
            df.loc[df["tempo"] <= 0, "tempo"] = pd.NA
            cap = safe_quantile_cap(df["tempo"], POLICY.tempo_cap_quantile)
            if cap is not None:
                df["tempo"] = df["tempo"].clip(upper=float(cap))

        # loudness in Range
        if "loudness" in df.columns:
            df["loudness"] = to_float(df["loudness"])
            lo, hi = POLICY.loudness_range
            df.loc[~df["loudness"].between(lo, hi), "loudness"] = pd.NA

        # ints
        for c in ["key", "mode", "time_signature"]:
            if c in df.columns:
                df[c] = to_int(df[c])

        if "key" in df.columns:
            lo, hi = POLICY.key_range
            df.loc[~df["key"].between(lo, hi), "key"] = pd.NA

        if "mode" in df.columns:
            lo, hi = POLICY.mode_range
            df.loc[~df["mode"].between(lo, hi), "mode"] = pd.NA

        # duration (in manchen Exports float)
        if "duration" in df.columns:
            df["duration"] = to_float(df["duration"])
            df.loc[df["duration"] <= 0, "duration"] = pd.NA

        df = keep_most_complete_row(df, ["id"])
        return df


class AlbumsCleaner(BaseCleaner):
    """Cleaning-Regeln für Tabelle `albums`."""
    name = "albums"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "id" not in df.columns:
            raise ValueError("albums: Missing primary key column 'id'.")

        df["id"] = norm_str(df["id"])
        df = df[df["id"].notna()]

        for c in ["name", "album_group", "album_type", "release_date"]:
            if c in df.columns:
                df[c] = norm_str(df[c])

        # Typen normalisieren (kleinschreibung)
        if "album_group" in df.columns:
            df["album_group"] = df["album_group"].str.lower()
        if "album_type" in df.columns:
            df["album_type"] = df["album_type"].str.lower()

        # popularity
        if "popularity" in df.columns:
            df["popularity"] = clip_series(to_int(df["popularity"]), POLICY.popularity_min, POLICY.popularity_max)

        # release_date parsed (wir behalten original + parsed)
        if "release_date" in df.columns:
            df["release_date_parsed"] = parse_release_date_universal(df["release_date"])

        df = keep_most_complete_row(df, ["id"])
        return df


class ArtistsCleaner(BaseCleaner):
    """Cleaning-Regeln für Tabelle `artists`."""
    name = "artists"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "id" not in df.columns:
            raise ValueError("artists: Missing primary key column 'id'.")

        df["id"] = norm_str(df["id"])
        df = df[df["id"].notna()]

        if "name" in df.columns:
            df["name"] = norm_str(df["name"])

        if "popularity" in df.columns:
            df["popularity"] = clip_series(to_int(df["popularity"]), POLICY.popularity_min, POLICY.popularity_max)

        # followers: nicht-negativ
        if "followers" in df.columns:
            df["followers"] = to_int(df["followers"])
            df["followers"] = df["followers"].clip(lower=0)

        df = keep_most_complete_row(df, ["id"])
        return df


class GenresCleaner(BaseCleaner):
    """Cleaning-Regeln für Tabelle `genres` (typischerweise nur id)."""
    name = "genres"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "id" not in df.columns:
            raise ValueError("genres: Missing primary key column 'id'.")

        df["id"] = norm_str(df["id"])
        df = df[df["id"].notna()].drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
        return df


class BridgeCleaner(BaseCleaner):
    """
    Generischer Cleaner für Bridge-Tabellen (Many-to-Many).

    Aufgaben:
    - Normalisiert Composite Keys (Strings)
    - Entfernt Zeilen mit NA in Keys
    - Entfernt Duplikate auf den Key-Kombinationen
    """
    def __init__(self, name: str, key_cols: List[str]):
        self.name = name
        self.key_cols = key_cols

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Keys müssen existieren
        missing_keys = [c for c in self.key_cols if c not in df.columns]
        if missing_keys:
            raise ValueError(f"{self.name}: Missing key columns {missing_keys}")

        for c in self.key_cols:
            df[c] = norm_str(df[c])

        df = df.dropna(subset=self.key_cols)
        df = df.drop_duplicates(subset=self.key_cols, keep="first").reset_index(drop=True)
        return df


def default_cleaners() -> List[BaseCleaner]:
    """
    Liefert die Standard-Cleaner in sinnvoller Reihenfolge.
    (Kann im Notebook genutzt werden, um eine Cleaning-Pipeline zu bauen.)
    """
    return [
        TracksCleaner(),
        AudioFeaturesCleaner(),
        AlbumsCleaner(),
        ArtistsCleaner(),
        GenresCleaner(),
        BridgeCleaner("r_albums_tracks", ["album_id", "track_id"]),
        BridgeCleaner("r_track_artist", ["track_id", "artist_id"]),
        BridgeCleaner("r_artist_genre", ["genre_id", "artist_id"]),
        BridgeCleaner("r_albums_artists", ["album_id", "artist_id"]),
    ]



def run_cleaning_pipeline(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Führt alle Standard-Cleaner in Reihenfolge aus.
    Es werden nur Tabellen gereinigt, die im raw-Dict existieren.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for cleaner in default_cleaners():
        name = cleaner.name
        if name in raw:
            cleaned[name] = cleaner.clean(raw[name])

    return cleaned



def id_set(tables: Dict[str, pd.DataFrame], table: str, col: str) -> Set[str]:
    """Liefert eindeutige IDs aus einer Tabelle, falls vorhanden – sonst leeres Set."""
    if table not in tables or col not in tables[table].columns:
        return set()
    return set(tables[table][col].dropna().unique())


def filter_fk(df: pd.DataFrame, col: str, allowed: Set[str]) -> pd.DataFrame:
    """Filtert DF auf Zeilen, deren FK in allowed liegt (NA wird entfernt, weil orphans)."""
    if df.empty or col not in df.columns or not allowed:
        return df
    return df[df[col].isin(allowed)].copy()


def clean_bridge_fks(
        tables: Dict[str, pd.DataFrame],
        fk_specs: Dict[str, Tuple[Tuple[str, Set[str]], ...]],
) -> Dict[str, pd.DataFrame]:
    """
    Entfernt Orphans in Bridge-Tabellen anhand FK-Spezifikation.
    fk_specs Beispiel:
      {"r_albums_tracks": (("album_id", album_ids), ("track_id", track_ids)), ...}
    """
    out = dict(tables)
    for bridge_name, fks in fk_specs.items():
        if bridge_name not in out:
            continue

        df = out[bridge_name]
        for col, allowed in fks:
            df = filter_fk(df, col, allowed)

        out[bridge_name] = df.reset_index(drop=True)

    return out


def enforce_optional_fk_as_na(
        tables: Dict[str, pd.DataFrame],
        table: str,
        fk_col: str,
        allowed_ids: Set[str],
) -> Dict[str, pd.DataFrame]:
    """
    Für optionale FKs: invalid -> NA (statt Zeile droppen).
    Beispiel: tracks.audio_feature_id
    """
    if table not in tables or fk_col not in tables[table].columns or not allowed_ids:
        return tables

    out = dict(tables)
    df = out[table].copy()

    bad = df[fk_col].notna() & ~df[fk_col].isin(allowed_ids)
    df.loc[bad, fk_col] = pd.NA

    out[table] = df
    return out

# =============================================================================
# Profiling Helpers (Specs + Batch-Profiling)
# =============================================================================


# Standard-Key-Spezifikation pro Tabelle (für Profiling/Reports)
PROFILE_SPECS: Dict[str, List[str]] = {
"tracks": ["track_id"], # Hinweis: Export kann auch "id" liefern -> handled in build_profiles()
"audio_features": ["id"],
"albums": ["id"],
"artists": ["id"],
"genres": ["id"],
"r_albums_tracks": ["album_id", "track_id"],
"r_track_artist": ["track_id", "artist_id"],
"r_artist_genre": ["genre_id", "artist_id"],
"r_albums_artists": ["album_id", "artist_id"],
}


def build_profiles(
tables: Dict[str, pd.DataFrame],
        specs=None,
    ) -> Dict[str, Dict[str, Any]]:
    """
    Erstellt profile_table() für alle vorhandenen Tabellen anhand einer Specs-Map.


    Spezialfall:
    - tracks kann im Raw-Export als 'id' statt 'track_id' kommen.
    Deshalb wird hier automatisch ['track_id'] genutzt, wenn vorhanden, sonst ['id'].
    """
    if specs is None:
        specs = PROFILE_SPECS

    out: Dict[str, Dict[str, Any]] = {}
    for name, key_cols in specs.items():
        if name not in tables:
            continue

    df = tables[name]
    # Sonderfall tracks: key abhängig von Spaltenname
    if name == "tracks":
        if "track_id" in df.columns:
            key_cols = ["track_id"]
    elif "id" in df.columns:
        key_cols = ["id"]

    out[name] = asdict(profile_table(df, name, key_cols=key_cols))
    return out

# =============================================================================
# Outlier / Rule Stage (Post-Cleaning)
# =============================================================================

@dataclass(frozen=True)
class OutlierPolicy:
    """
    Policy für Outlier-Regeln (separat vom Schema-Cleaning).
    """
    q_low: float = 0.001
    q_high: float = 0.999
    track_number_max: int = 200
    disc_number_max: int = 10
    valid_time_signatures: Tuple[int, ...] = (3, 4, 5)
    release_year_min: int = 1900
    release_year_max: int = 2035
    loudness_hard_range: Tuple[float, float] = (-60.0, 5.0)
    loudness_very_low: float = -40.0
    instrumental_threshold: float = 0.5
    speechiness_high_q: float = 0.90

OUTLIER_POLICY = OutlierPolicy()


def quantile_cap(series: pd.Series, q_low: float, q_high: float) -> Tuple[float, float]:
    """Berechnet Quantil-Grenzen robust (liefert NaN, wenn keine Daten)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    return (float(s.quantile(q_low)), float(s.quantile(q_high)))


class BaseRuleStage:
    """Basisklasse für Regel-Stufen nach dem Cleaning."""
    name: str
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TracksOutlierRules(BaseRuleStage):
    name = "tracks"
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df.copy()

        if "popularity" in t.columns:
            t["popularity"] = pd.to_numeric(t["popularity"], errors="coerce").clip(0, 100)

        if "duration" in t.columns:
            t["duration"] = pd.to_numeric(t["duration"], errors="coerce")
            t.loc[t["duration"] <= 0, "duration"] = pd.NA

            _, hi = quantile_cap(t["duration"], OUTLIER_POLICY.q_low, OUTLIER_POLICY.q_high)
            t["is_long_track"] = (t["duration"] > hi).astype("int8") if not np.isnan(hi) else 0
            if not np.isnan(hi):
                t["duration"] = t["duration"].clip(upper=int(hi))

        if "track_number" in t.columns:
            t["track_number"] = pd.to_numeric(t["track_number"], errors="coerce")
            t.loc[t["track_number"] <= 0, "track_number"] = pd.NA
            t["is_tracknum_extreme"] = (t["track_number"] > OUTLIER_POLICY.track_number_max).astype("int8")
            t.loc[t["track_number"] > OUTLIER_POLICY.track_number_max, "track_number"] = pd.NA

        if "disc_number" in t.columns:
            t["disc_number"] = pd.to_numeric(t["disc_number"], errors="coerce")
            t.loc[t["disc_number"] <= 0, "disc_number"] = pd.NA
            t["is_multidisc"] = (t["disc_number"] > 1).astype("int8")
            t["is_disc_extreme"] = (t["disc_number"] > OUTLIER_POLICY.disc_number_max).astype("int8")
            t.loc[t["disc_number"] > OUTLIER_POLICY.disc_number_max, "disc_number"] = pd.NA

        return t


class AudioFeaturesOutlierRules(BaseRuleStage):
    name = "audio_features"
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        a = df.copy()

        if "time_signature" in a.columns:
            a["time_signature"] = pd.to_numeric(a["time_signature"], errors="coerce")
            valid = set(OUTLIER_POLICY.valid_time_signatures)
            a["is_time_signature_rare"] = (~a["time_signature"].isin(list(valid))).astype("int8")
            a.loc[~a["time_signature"].isin(list(valid)), "time_signature"] = pd.NA

        if "tempo" in a.columns:
            a["tempo"] = pd.to_numeric(a["tempo"], errors="coerce")
            a.loc[a["tempo"] <= 0, "tempo"] = pd.NA
            lo, hi = quantile_cap(a["tempo"], OUTLIER_POLICY.q_low, OUTLIER_POLICY.q_high)
            a["is_tempo_extreme"] = ((a["tempo"] < lo) | (a["tempo"] > hi)).astype("int8") if not np.isnan(hi) else 0
            if not np.isnan(hi):
                a["tempo"] = a["tempo"].clip(upper=hi)

        if "loudness" in a.columns:
            a["loudness"] = pd.to_numeric(a["loudness"], errors="coerce")
            lo, hi = OUTLIER_POLICY.loudness_hard_range
            a.loc[~a["loudness"].between(lo, hi), "loudness"] = pd.NA
            a["is_loudness_very_low"] = (a["loudness"] < OUTLIER_POLICY.loudness_very_low).astype("int8")

        if "duration" in a.columns:
            a["duration"] = pd.to_numeric(a["duration"], errors="coerce")
            a.loc[a["duration"] <= 0, "duration"] = pd.NA
            _, hi = quantile_cap(a["duration"], OUTLIER_POLICY.q_low, OUTLIER_POLICY.q_high)
            a["is_af_long"] = (a["duration"] > hi).astype("int8") if not np.isnan(hi) else 0
            if not np.isnan(hi):
                a["duration"] = a["duration"].clip(upper=hi)

        if "speechiness" in a.columns:
            a["speechiness"] = pd.to_numeric(a["speechiness"], errors="coerce").clip(0, 1)
            if a["speechiness"].notna().any():
                thr = a["speechiness"].dropna().quantile(OUTLIER_POLICY.speechiness_high_q)
                a["is_high_speech"] = (a["speechiness"] >= thr).astype("int8")
            else:
                a["is_high_speech"] = 0

        if "instrumentalness" in a.columns:
            a["instrumentalness"] = pd.to_numeric(a["instrumentalness"], errors="coerce").clip(0, 1)
            a["is_instrumental"] = (a["instrumentalness"] >= OUTLIER_POLICY.instrumental_threshold).astype("int8")

        if "key" in a.columns:
            a["key"] = pd.to_numeric(a["key"], errors="coerce")
            a.loc[~a["key"].between(0, 11), "key"] = pd.NA

        if "mode" in a.columns:
            a["mode"] = pd.to_numeric(a["mode"], errors="coerce")
            a.loc[~a["mode"].between(0, 1), "mode"] = pd.NA

        return a


class ArtistsOutlierRules(BaseRuleStage):
    name = "artists"
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ar = df.copy()

        if "followers" in ar.columns:
            ar["followers"] = pd.to_numeric(ar["followers"], errors="coerce")
            ar.loc[ar["followers"] < 0, "followers"] = pd.NA
            _, hi = quantile_cap(ar["followers"], OUTLIER_POLICY.q_low, OUTLIER_POLICY.q_high)
            ar["is_followers_extreme"] = (ar["followers"] > hi).astype("int8") if not np.isnan(hi) else 0
            ar["followers_log1p"] = np.log1p(ar["followers"].fillna(0)).astype("float64")

        if "popularity" in ar.columns:
            ar["popularity"] = pd.to_numeric(ar["popularity"], errors="coerce").clip(0, 100)

        return ar


class AlbumsOutlierRules(BaseRuleStage):
    name = "albums"
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        al = df.copy()

        if "release_date_parsed" in al.columns:
            years = pd.to_datetime(al["release_date_parsed"], errors="coerce").dt.year
            al["is_release_year_invalid"] = ((years < OUTLIER_POLICY.release_year_min) | (years > OUTLIER_POLICY.release_year_max)).astype("int8")
            al.loc[(years < OUTLIER_POLICY.release_year_min) | (years > OUTLIER_POLICY.release_year_max), "release_date_parsed"] = pd.NaT
            al["release_year"] = pd.to_datetime(al["release_date_parsed"], errors="coerce").dt.year.astype("Int64")

        if "popularity" in al.columns:
            al["popularity"] = pd.to_numeric(al["popularity"], errors="coerce").clip(0, 100)

        return al


def default_rule_stages() -> List[BaseRuleStage]:
    """Standard-Regelstufen nach dem Cleaning."""
    return [
        TracksOutlierRules(),
        AudioFeaturesOutlierRules(),
        ArtistsOutlierRules(),
        AlbumsOutlierRules(),
    ]


def apply_rule_stages(
    tables: Dict[str, pd.DataFrame],
    stages: Optional[List[BaseRuleStage]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Wendet Rule-Stages auf vorhandene Tabellen an (nur wenn im Dict vorhanden).
    """
    if stages is None:
        stages = default_rule_stages()

    out = dict(tables)
    for st in stages:
        if st.name in out:
            out[st.name] = st.apply(out[st.name])
    return out

def run_quality_gates(
    cleaned: Dict[str, pd.DataFrame],
    id_sets: Optional[Dict[str, set]] = None,
    policy: CleaningPolicy = POLICY,
):
    """
    Führt zentrale Quality Gates aus:
    - PKs: non-null + unique
    - Ranges: popularity, duration, audio_01_cols
    - Bridges: referential integrity (optional, wenn id_sets übergeben)
    """
    # --- PK Gates ---
    pk_specs = {
        "tracks": ("track_id",),
        "audio_features": ("id",),
        "albums": ("id",),
        "artists": ("id",),
        "genres": ("id",),
    }

    for tbl, (pk,) in pk_specs.items():
        if tbl not in cleaned or pk not in cleaned[tbl].columns:
            continue
        assert_gate(cleaned[tbl][pk].notna().all(), f"{tbl}.{pk} contains NA")
        assert_gate(cleaned[tbl][pk].is_unique, f"{tbl}.{pk} not unique")

    # --- Range Gates ---
    if "tracks" in cleaned and "popularity" in cleaned["tracks"].columns:
        assert_gate(cleaned["tracks"]["popularity"].dropna().between(0, 100).all(), "tracks.popularity out of [0,100]")

    if "tracks" in cleaned and "duration" in cleaned["tracks"].columns:
        assert_gate((cleaned["tracks"]["duration"].dropna() > 0).all(), "tracks.duration has non-positive values")

    if "audio_features" in cleaned:
        for c in policy.audio_01_cols:
            if c in cleaned["audio_features"].columns:
                assert_gate(cleaned["audio_features"][c].dropna().between(0.0, 1.0).all(), f"audio_features.{c} out of [0,1]")

    # --- Bridge FK Gates (optional) ---
    if id_sets:
        fk_specs = {
            "r_albums_tracks":  (("album_id", "albums"), ("track_id", "tracks")),
            "r_track_artist":   (("track_id", "tracks"), ("artist_id", "artists")),
            "r_artist_genre":   (("genre_id", "genres"), ("artist_id", "artists")),
            "r_albums_artists": (("album_id", "albums"), ("artist_id", "artists")),
        }

        for bridge, refs in fk_specs.items():
            if bridge not in cleaned:
                continue
            df = cleaned[bridge]
            for col, target in refs:
                allowed = id_sets.get(target, set())
                if col in df.columns and allowed:
                    assert_gate(df[col].isin(allowed).all(), f"{bridge} has invalid {col}")

    return True

# =============================================================================
# Export / Reporting
# =============================================================================

import json
from pathlib import Path

def save_clean_layer_parquet(
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Speichert alle Tabellen als Parquet nach out_dir/<name>.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def compute_rowcount_delta(
    before: Dict[str, pd.DataFrame],
    after: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, int]]:
    """Berechnet row counts before/after sowie delta pro Tabelle."""
    out: Dict[str, Dict[str, int]] = {}
    names = sorted(set(before.keys()) | set(after.keys()))
    for name in names:
        b = int(before[name].shape[0]) if name in before else 0
        a = int(after[name].shape[0]) if name in after else 0
        out[name] = {"before": b, "after": a, "delta": a - b}
    return out


def write_cleaning_report(
    report_path: Path,
    profiles_before: Optional[Dict[str, Any]] = None,
    profiles_after: Optional[Dict[str, Any]] = None,
    rowcount_delta: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Schreibt einen Cleaning-Report als JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "run_meta": run_meta or {},
        "profiles_before": profiles_before or {},
        "profiles_after": profiles_after or {},
        "rowcount_delta": rowcount_delta or {},
        "notes": notes or {},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")