# utils/features/engineering.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Literal, Dict, List

import numpy as np
import pandas as pd

from ..features.numeric_transform import log1p_numeric
from ..features.text_features import  safe_len_series, safe_word_count_series
from ..features.time_features import  col_or_na
from ..data.parsing import  ensure_list_column

# =============================================================================
# Gemeinsames Interface
# =============================================================================

class EntityEngineer(Protocol):
    """Interface: jeder Engineer nimmt ein DataFrame und gibt ein DataFrame zurück."""
    name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


# =============================================================================
# Track Engineer
# =============================================================================

@dataclass(frozen=True)
class TrackEngineerConfig:
    """Konfiguration für Track-Feature Engineering."""
    duration_cols: tuple[str, ...] = ("duration", "duration_ms")
    instrumental_threshold: float = 0.5
    tempo_fast_threshold: float = 160.0
    loudness_very_low_threshold: float = -40.0
    speechiness_high_q: float = 0.90

    # Optional: zusätzliche Heuristiken
    max_track_number_valid: int = 200
    max_disc_number_valid: int = 10


class TrackEngineer:
    """
    Feature Engineering für Track-Level Dataset.
    Erwartet ein bereits zusammengejointes track_df (aus TrackDatasetBuilder).
    """
    name = "track"

    def __init__(self, config: TrackEngineerConfig = TrackEngineerConfig()):
        self.cfg = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Blöcke bewusst getrennt → leichter debuggen/erweitern
        df = self._add_flags(df)
        df = self._add_text_features(df)
        df = self._add_numeric_transforms(df)
        df = self._add_audio_features_derived(df)
        df = self._add_genre_features(df)
        df = self._add_time_segments(df)
        df = self._add_structural_sanity_flags(df)

        return df

    # -------------------------
    # Feature Blocks
    # -------------------------

    def _add_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Einfache binäre Flags."""
        # Preview vorhanden? (falls preview_url im Cleaning gedroppt wurde, existiert has_preview evtl. schon)
        if "has_preview" not in df.columns:
            df["has_preview"] = col_or_na(df, "preview_url").notna().astype("int8")

        # Explicit Flag (falls vorhanden)
        if "explicit" in df.columns:
            df["is_explicit"] = df["explicit"].astype("boolean").fillna(False).astype("int8")

        # Audio-Features vorhanden?
        df["has_audio_features"] = col_or_na(df, "audio_feature_id").notna().astype("int8")

        # Multi-Artist Kollaboration?
        if "n_artists" in df.columns:
            df["is_collab"] = (pd.to_numeric(df["n_artists"], errors="coerce").fillna(0) > 1).astype("int8")

        return df

    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Textbasierte Features aus Track-Namen."""
        df["name_len"] = safe_len_series(col_or_na(df, "name"))
        df["name_words"] = safe_word_count_series(col_or_na(df, "name"))
        return df

    def _add_numeric_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numerische Transformationen / stabile Skalen."""
        # Log(Duration)
        dur_col = next((c for c in self.cfg.duration_cols if c in df.columns), None)
        df["log_duration"] = log1p_numeric(df[dur_col]) if dur_col else pd.Series(np.nan, index=df.index)

        # Artist follower transforms (falls vorhanden)
        df["log_artist_followers_max"] = log1p_numeric(col_or_na(df, "artist_followers_max"))
        df["log_artist_followers_mean"] = log1p_numeric(col_or_na(df, "artist_followers_mean"))

        # Popularity: stabil als numeric
        if "popularity" in df.columns:
            df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

        if "album_popularity" in df.columns:
            df["album_popularity"] = pd.to_numeric(df["album_popularity"], errors="coerce")

        # Differenz kann Signal sein (Track vs Album)
        if "popularity" in df.columns and "album_popularity" in df.columns:
            df["popularity_vs_album"] = df["popularity"] - df["album_popularity"]

        # Dispersion: wie „ungleich“ sind Artists (max - mean)
        if "artist_followers_max" in df.columns and "artist_followers_mean" in df.columns:
            df["artist_followers_dispersion"] = (
                pd.to_numeric(df["artist_followers_max"], errors="coerce")
                - pd.to_numeric(df["artist_followers_mean"], errors="coerce")
            )

        if "artist_popularity_max" in df.columns and "artist_popularity_mean" in df.columns:
            df["artist_popularity_dispersion"] = (
                pd.to_numeric(df["artist_popularity_max"], errors="coerce")
                - pd.to_numeric(df["artist_popularity_mean"], errors="coerce")
            )

        return df

    def _add_audio_features_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abgeleitete Audio-Features (Flags + interpretable Kombis)."""
        # Tempo Flag
        if "tempo" in df.columns:
            tempo = pd.to_numeric(df["tempo"], errors="coerce")
            df["is_fast_tempo"] = (tempo >= self.cfg.tempo_fast_threshold).astype("int8")

        # Loudness Flag
        if "loudness" in df.columns:
            loud = pd.to_numeric(df["loudness"], errors="coerce")
            df["is_very_quiet"] = (loud < self.cfg.loudness_very_low_threshold).astype("int8")

        # Instrumental
        if "instrumentalness" in df.columns:
            inst = pd.to_numeric(df["instrumentalness"], errors="coerce")
            df["is_instrumental"] = (inst >= self.cfg.instrumental_threshold).astype("int8")

        # Speechiness high-quantile Flag (pro Sample!)
        if "speechiness" in df.columns:
            sp = pd.to_numeric(df["speechiness"], errors="coerce")
            if sp.notna().any():
                thr = sp.dropna().quantile(self.cfg.speechiness_high_q)
                df["is_high_speech"] = (sp >= thr).astype("int8")
            else:
                df["is_high_speech"] = 0

        # Mood Quadrants (Energy x Valence) → gut interpretierbar
        if "energy" in df.columns and "valence" in df.columns:
            e = pd.to_numeric(df["energy"], errors="coerce")
            v = pd.to_numeric(df["valence"], errors="coerce")
            df["mood_happy_energetic"] = ((e >= 0.5) & (v >= 0.5)).astype("int8")
            df["mood_angry_energetic"] = ((e >= 0.5) & (v < 0.5)).astype("int8")
            df["mood_calm_happy"] = ((e < 0.5) & (v >= 0.5)).astype("int8")
            df["mood_sad_calm"] = ((e < 0.5) & (v < 0.5)).astype("int8")

        return df

    def _add_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genre Features aus track_genres (Listen-Spalte)."""
        genres = ensure_list_column(col_or_na(df, "track_genres"))
        df["n_genres"] = genres.apply(len).astype("int16")
        df["has_genre"] = (df["n_genres"] > 0).astype("int8")

        # Simple Heuristik: erstes Genre als Main-Genre-ID
        df["main_genre_id"] = genres.apply(lambda xs: xs[0] if len(xs) else pd.NA).astype("string")
        return df

    def _add_time_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) baue release_date_parsed (Album first, Track fallback)
        if "release_date_parsed" not in df.columns:
            a = col_or_na(df, "album_release_date_parsed").replace("", pd.NA)
            b = col_or_na(df, "track_release_date_parsed").replace("", pd.NA)

            a = pd.to_datetime(a, errors="coerce")
            b = pd.to_datetime(b, errors="coerce")
            df["release_date_parsed"] = a.combine_first(b)

        # 2) release_year aus release_date_parsed
        if "release_year" not in df.columns:
            df["release_year"] = pd.to_datetime(df["release_date_parsed"], errors="coerce").dt.year

        # 3) Segmente (NA-safe)
        year = pd.to_numeric(df["release_year"], errors="coerce")
        df["is_modern_release"] = (year >= 2015).fillna(False).astype("int8")
        df["is_old_release"] = (year < 1990).fillna(False).astype("int8")
        df["is_release_year_missing"] = year.isna().astype("int8")

        return df

    def _add_structural_sanity_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Struktur-basierte Flags (Track/Disc Numbers).
        NICHT droppen → nur markieren / invalid ggf. NA.
        """
        df = df.copy()

        if "track_number" in df.columns:
            tn = pd.to_numeric(df["track_number"], errors="coerce")

            extreme = tn.gt(self.cfg.max_track_number_valid)  # kann <NA> enthalten
            df["is_tracknum_extreme"] = np.where(extreme.fillna(False), 1, 0).astype("int8")

            # invalid -> NA
            df.loc[extreme.fillna(False), "track_number"] = pd.NA
            df["is_tracknum_missing"] = tn.isna().astype("int8")  # optional

        if "disc_number" in df.columns:
            dn = pd.to_numeric(df["disc_number"], errors="coerce")

            multidisc = dn.gt(1)
            df["is_multidisc"] = np.where(multidisc.fillna(False), 1, 0).astype("int8")

            extreme = dn.gt(self.cfg.max_disc_number_valid)
            df["is_disc_extreme"] = np.where(extreme.fillna(False), 1, 0).astype("int8")

            df.loc[extreme.fillna(False), "disc_number"] = pd.NA
            df["is_disc_missing"] = dn.isna().astype("int8")  # optional

        return df


# =============================================================================
# Album Engineer (für später)
# =============================================================================

@dataclass(frozen=True)
class AlbumEngineerConfig:
    """
    Advanced Album Feature Engineering.
    Erwartet album_df aus AlbumDatasetBuilder (mit n_tracks, album_mean_* und optional artist aggs).
    """
    # Core
    add_text_features: bool = True
    add_size_segments: bool = True
    add_release_recency: bool = True
    add_audio_signature_features: bool = True
    add_artist_power_features: bool = True
    add_genre_features: bool = True

    # Size thresholds
    single_max_tracks: int = 2
    ep_max_tracks: int = 7
    album_min_tracks: int = 8
    mega_album_min_tracks: int = 30

    # Recency: "modern era" (Spotify)
    modern_year_threshold: int = 2015
    old_year_threshold: int = 1990

    # Audio signature columns prefix in album_df
    audio_mean_prefix: str = "album_mean_"

    # For “cohesion”: if you also store std later, you can extend
    # currently only mean-based features.

    # dtype for flags
    flag_dtype: str = "int8"


class AlbumEngineer:
    """
    Engineering auf Album-Level.
    Macht KEINE Joins/Aggregationen mehr (das macht der Builder).
    """
    name = "album_engineer"

    def __init__(self, config: AlbumEngineerConfig = AlbumEngineerConfig()):
        self.cfg = config
        self.created_features_: List[str] = []

    def transform(self, df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
        df = df.copy()
        before_cols = set(df.columns)

        df = self._ensure_types(df)

        if self.cfg.add_text_features:
            df = self._add_text_features(df)

        if self.cfg.add_size_segments:
            df = self._add_size_segments(df)

        if self.cfg.add_release_recency:
            df = self._add_release_recency(df)

        if self.cfg.add_audio_signature_features:
            df = self._add_audio_signature_features(df)

        if self.cfg.add_artist_power_features:
            df = self._add_artist_power_features(df)

        if self.cfg.add_genre_features:
            df = self._add_genre_features(df)

        self.created_features_ = sorted(list(set(df.columns) - before_cols))
        if verbose:
            print(f" AlbumEngineer created {len(self.created_features_)} features")
            print("   ->", self.created_features_[:30], "..." if len(self.created_features_) > 30 else "")

        return df

    # ---------------------------------------------------------------------
    # Blocks
    # ---------------------------------------------------------------------

    def _ensure_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # numeric safety
        if "n_tracks" in df.columns:
            df["n_tracks"] = pd.to_numeric(df["n_tracks"], errors="coerce")

        # release_year might already exist from add_release_time_features
        if "release_year" in df.columns:
            df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

        # list safety
        if "album_genres" in df.columns:
            df["album_genres"] = ensure_list_column(col_or_na(df, "album_genres"))

        return df

    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # simple but useful
        df["album_name_len"] = safe_len_series(col_or_na(df, "name"))
        df["album_name_words"] = safe_word_count_series(col_or_na(df, "name"))

        # “special edition / deluxe” heuristics (helps to separate re-releases)
        name = col_or_na(df, "name").astype("string")
        lowered = name.str.lower()

        df["is_deluxe"] = lowered.str.contains("deluxe", na=False).astype(self.cfg.flag_dtype)
        df["is_remaster"] = lowered.str.contains("remaster|remastered", na=False).astype(self.cfg.flag_dtype)
        df["is_live_album"] = lowered.str.contains(r"\blive\b", na=False).astype(self.cfg.flag_dtype)
        df["is_compilation"] = lowered.str.contains("compilation|greatest hits|best of", na=False).astype(self.cfg.flag_dtype)

        return df

    def _add_size_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segmentiert nach Album-Größe.
        Sehr hilfreich, weil Singles/EPs/LPs strukturell andere Popularity-Profile haben.
        """
        n = col_or_na(df, "n_tracks")
        n = pd.to_numeric(n, errors="coerce")

        df["log_n_tracks"] = log1p_numeric(n)

        df["is_single"] = (n <= self.cfg.single_max_tracks).astype(self.cfg.flag_dtype)
        df["is_ep"] = ((n > self.cfg.single_max_tracks) & (n <= self.cfg.ep_max_tracks)).astype(self.cfg.flag_dtype)
        df["is_lp"] = (n >= self.cfg.album_min_tracks).astype(self.cfg.flag_dtype)
        df["is_mega_album"] = (n >= self.cfg.mega_album_min_tracks).astype(self.cfg.flag_dtype)

        # “unknown size” flag
        df["is_n_tracks_missing"] = n.isna().astype(self.cfg.flag_dtype)

        return df

    def _add_release_recency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recency / Era features.
        Robust gegen NA in release_year.
        """
        if "release_year" not in df.columns:
            return df

        df = df.copy()
        y = pd.to_numeric(df["release_year"], errors="coerce")

        modern = y.ge(self.cfg.modern_year_threshold)
        old = y.lt(self.cfg.old_year_threshold)

        df["is_modern_release"] = np.where(modern.fillna(False), 1, 0).astype(self.cfg.flag_dtype)
        df["is_old_release"] = np.where(old.fillna(False), 1, 0).astype(self.cfg.flag_dtype)
        df["is_release_year_missing"] = y.isna().astype(self.cfg.flag_dtype)  # optional, aber hilfreich

        # decade already might exist (release_decade), but ensure stable
        if "release_decade" in df.columns:
            df["release_decade"] = pd.to_numeric(df["release_decade"], errors="coerce").astype("Int64")
        else:
            df["release_decade"] = (np.floor(y / 10) * 10).astype("Int64")

        # recency score normalize [0,1]
        y_valid = y.dropna()
        if len(y_valid) > 0:
            y_min, y_max = float(y_valid.min()), float(y_valid.max())
            if y_max > y_min:
                df["release_year_norm"] = ((y - y_min) / (y_max - y_min)).astype("float64")
            else:
                df["release_year_norm"] = 0.0
        else:
            df["release_year_norm"] = np.nan

        return df
    def _add_audio_signature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nutzt album_mean_* Features (vom Builder) und erzeugt:
        - mood quadrants
        - high/low tempo flags
        - “dance_energy” interactions
        - loudness normalization flags
        """
        p = self.cfg.audio_mean_prefix
        flag = self.cfg.flag_dtype

        # helper to fetch mean cols
        def c(name: str) -> Optional[pd.Series]:
            col = f"{p}{name}"
            return df[col] if col in df.columns else None

        energy = c("energy")
        valence = c("valence")
        dance = c("danceability")
        tempo = c("tempo")
        loud = c("loudness")
        speech = c("speechiness")
        instr = c("instrumentalness")

        # mood quadrants
        if energy is not None and valence is not None:
            e = pd.to_numeric(energy, errors="coerce")
            v = pd.to_numeric(valence, errors="coerce")
            df["album_mood_happy_energetic"] = ((e >= 0.5) & (v >= 0.5)).astype(flag)
            df["album_mood_angry_energetic"] = ((e >= 0.5) & (v < 0.5)).astype(flag)
            df["album_mood_calm_happy"] = ((e < 0.5) & (v >= 0.5)).astype(flag)
            df["album_mood_sad_calm"] = ((e < 0.5) & (v < 0.5)).astype(flag)

        # interactions (continuous)
        if energy is not None and dance is not None:
            e = pd.to_numeric(energy, errors="coerce")
            d = pd.to_numeric(dance, errors="coerce")
            df["album_dance_x_energy"] = (d * e).astype("float64")

        # tempo segmentation (relative within dataset)
        if tempo is not None:
            t = pd.to_numeric(tempo, errors="coerce")
            if t.notna().any():
                q25, q75 = t.dropna().quantile(0.25), t.dropna().quantile(0.75)
                df["album_is_slow_tempo"] = (t <= q25).astype(flag)
                df["album_is_fast_tempo"] = (t >= q75).astype(flag)

        # loudness “quiet/loud” relative
        if loud is not None:
            l = pd.to_numeric(loud, errors="coerce")
            if l.notna().any():
                med = l.dropna().median()
                df["album_is_quiet"] = (l < med).astype(flag)

        # speechiness/instrumentalness high flags (relative)
        if speech is not None:
            s = pd.to_numeric(speech, errors="coerce")
            if s.notna().any():
                thr = s.dropna().quantile(0.90)
                df["album_is_high_speech"] = (s >= thr).astype(flag)

        if instr is not None:
            ins = pd.to_numeric(instr, errors="coerce")
            df["album_is_instrumental_heavy"] = (ins >= 0.5).astype(flag)

        return df

    def _add_artist_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        flag = self.cfg.flag_dtype
        df = df.copy()

        # n_album_artists -> collab flag (NA-safe)
        if "n_album_artists" in df.columns:
            n = pd.to_numeric(df["n_album_artists"], errors="coerce").fillna(0)
            df["album_is_collab"] = np.where(n > 1, 1, 0).astype(flag)

        if "album_artist_followers_mean" in df.columns:
            df["album_artist_followers_mean_log1p"] = log1p_numeric(col_or_na(df, "album_artist_followers_mean"))
        if "album_artist_followers_max" in df.columns:
            df["album_artist_followers_max_log1p"] = log1p_numeric(col_or_na(df, "album_artist_followers_max"))

        # gaps (float -> ok mit NA)
        if "album_artist_followers_max" in df.columns and "album_artist_followers_mean" in df.columns:
            mx = pd.to_numeric(df["album_artist_followers_max"], errors="coerce")
            me = pd.to_numeric(df["album_artist_followers_mean"], errors="coerce")
            df["album_artist_followers_gap"] = (mx - me).astype("float64")

        if "album_artist_popularity_max" in df.columns and "album_artist_popularity_mean" in df.columns:
            mx = pd.to_numeric(df["album_artist_popularity_max"], errors="coerce")
            me = pd.to_numeric(df["album_artist_popularity_mean"], errors="coerce")
            df["album_artist_popularity_gap"] = (mx - me).astype("float64")

        # headliner flag (NA-safe)
        if "album_artist_popularity_max" in df.columns:
            ap = pd.to_numeric(df["album_artist_popularity_max"], errors="coerce")
            if ap.notna().any():
                thr = ap.dropna().quantile(0.90)
                cond = ap.ge(thr)  # contains NA if ap is NA
                df["album_has_headliner_artist"] = np.where(cond.fillna(False), 1, 0).astype(flag)
            else:
                df["album_has_headliner_artist"] = 0  # all missing -> no headliner signal

        return df

    def _add_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genre list features:
        - count
        - has_genre
        - main_genre_id heuristic
        """
        flag = self.cfg.flag_dtype

        if "album_genres" not in df.columns:
            return df

        genres = ensure_list_column(col_or_na(df, "album_genres"))
        df["n_album_genres"] = genres.apply(len).astype("int16")
        df["album_has_genre"] = (df["n_album_genres"] > 0).astype(flag)

        # simple main genre heuristic
        df["album_main_genre_id"] = genres.apply(lambda xs: xs[0] if len(xs) else pd.NA).astype("string")

        # “multi-genre” flag (broadness)
        df["album_is_multi_genre"] = (df["n_album_genres"] >= 3).astype(flag)

        return df


# =============================================================================
# Artist Engineer (für später)
# =============================================================================
@dataclass(frozen=True)
class ArtistEngineerConfig:
    """
    Feature Engineering für Artist-Level Dataset.
    Erwartet artist_df aus ArtistDatasetBuilder (Option A).
    """
    # blocks
    add_numeric_safety: bool = True
    add_heavy_tail_logs: bool = True
    add_popularity_features: bool = True
    add_explicit_features: bool = True
    add_style_features: bool = True
    add_genre_features: bool = True

    # thresholds / quantiles (relativ im aktuellen df)
    headliner_pop_q: float = 0.90
    superstar_followers_q: float = 0.90

    # dtype policy
    flag_dtype: str = "int8"


class ArtistEngineer:
    name = "artist_engineer"

    def __init__(self, config: ArtistEngineerConfig = ArtistEngineerConfig()):
        self.cfg = config
        self.created_features_: List[str] = []

    def transform(self, df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
        df = df.copy()
        before = set(df.columns)

        if self.cfg.add_numeric_safety:
            df = self._numeric_safety(df)

        if self.cfg.add_heavy_tail_logs:
            df = self._heavy_tail_logs(df)

        if self.cfg.add_popularity_features:
            df = self._popularity_features(df)

        if self.cfg.add_explicit_features:
            df = self._explicit_features(df)

        if self.cfg.add_style_features:
            df = self._style_features(df)

        if self.cfg.add_genre_features:
            df = self._genre_features(df)

        self.created_features_ = sorted(list(set(df.columns) - before))
        if verbose:
            print(f"ArtistEngineer created {len(self.created_features_)} features")
            print("   ->", self.created_features_[:30], "..." if len(self.created_features_) > 30 else "")

        return df

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _numeric_safety(self, df: pd.DataFrame) -> pd.DataFrame:
        # core numeric columns often used
        for c in [
            "followers", "popularity", "n_tracks",
            "track_pop_mean", "explicit_rate"
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # mean audio cols from builder: mean_energy, mean_danceability, ...
        for c in df.columns:
            if c.startswith("mean_"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # list safety
        if "artist_genres" in df.columns:
            df["artist_genres"] = ensure_list_column(col_or_na(df, "artist_genres"))

        return df

    def _heavy_tail_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        # stable logs for skewed distributions
        if "followers" in df.columns:
            df["followers_log1p"] = log1p_numeric(col_or_na(df, "followers"))
        if "n_tracks" in df.columns:
            df["n_tracks_log1p"] = log1p_numeric(col_or_na(df, "n_tracks"))

        # “followers per track” (productivity vs reach) — robust ratio
        if "followers" in df.columns and "n_tracks" in df.columns:
            f = pd.to_numeric(df["followers"], errors="coerce")
            n = pd.to_numeric(df["n_tracks"], errors="coerce")
            df["followers_per_track"] = (f / (n.replace(0, np.nan))).astype("float64")
            df["followers_per_track_log1p"] = log1p_numeric(col_or_na(df, "followers_per_track"))

        return df

    def _popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        flag = self.cfg.flag_dtype

        # base popularity numeric
        if "popularity" in df.columns:
            pop = pd.to_numeric(df["popularity"], errors="coerce")

            # relative flags (within df)
            if pop.notna().any():
                thr = pop.dropna().quantile(self.cfg.headliner_pop_q)
                df["is_headliner_popularity"] = (pop >= thr).astype(flag)
            else:
                df["is_headliner_popularity"] = 0

        # compare artist popularity vs avg track popularity (if available)
        if "popularity" in df.columns and "track_pop_mean" in df.columns:
            ap = pd.to_numeric(df["popularity"], errors="coerce")
            tp = pd.to_numeric(df["track_pop_mean"], errors="coerce")
            df["artist_vs_track_pop_gap"] = (ap - tp).astype("float64")

        return df

    def _explicit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        flag = self.cfg.flag_dtype

        if "explicit_rate" in df.columns:
            er = pd.to_numeric(df["explicit_rate"], errors="coerce")

            # interpretierbare flags
            df["is_mostly_explicit"] = (er >= 0.5).astype(flag)
            df["is_never_explicit"] = (er == 0).astype(flag)
            df["explicit_rate_filled"] = er.fillna(0).astype("float64")

        return df

    def _style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Style/Signature derived features aus mean_* Audio.
        Alles optional — nur wenn Spalten existieren.
        """
        flag = self.cfg.flag_dtype

        def get(name: str) -> Optional[pd.Series]:
            c = f"mean_{name}"
            return df[c] if c in df.columns else None

        energy = get("energy")
        valence = get("valence")
        dance = get("danceability")
        tempo = get("tempo")
        loud = get("loudness")
        speech = get("speechiness")
        instr = get("instrumentalness")
        acoustic = get("acousticness")

        # mood quadrants (energy x valence)
        if energy is not None and valence is not None:
            e = pd.to_numeric(energy, errors="coerce")
            v = pd.to_numeric(valence, errors="coerce")
            df["artist_mood_happy_energetic"] = ((e >= 0.5) & (v >= 0.5)).astype(flag)
            df["artist_mood_angry_energetic"] = ((e >= 0.5) & (v < 0.5)).astype(flag)
            df["artist_mood_calm_happy"] = ((e < 0.5) & (v >= 0.5)).astype(flag)
            df["artist_mood_sad_calm"] = ((e < 0.5) & (v < 0.5)).astype(flag)

        # interaction: dance * energy (dancefloor intensity)
        if dance is not None and energy is not None:
            d = pd.to_numeric(dance, errors="coerce")
            e = pd.to_numeric(energy, errors="coerce")
            df["dance_x_energy"] = (d * e).astype("float64")

        # relative tempo flags (dataset-relative)
        if tempo is not None:
            t = pd.to_numeric(tempo, errors="coerce")
            if t.notna().any():
                q25, q75 = t.dropna().quantile(0.25), t.dropna().quantile(0.75)
                df["is_slow_tempo_artist"] = (t <= q25).astype(flag)
                df["is_fast_tempo_artist"] = (t >= q75).astype(flag)

        # acoustic vs electronic proxy
        if acoustic is not None and energy is not None:
            a = pd.to_numeric(acoustic, errors="coerce")
            e = pd.to_numeric(energy, errors="coerce")
            df["acoustic_minus_energy"] = (a - e).astype("float64")

        # speech/instrumental “high” flags
        if speech is not None:
            s = pd.to_numeric(speech, errors="coerce")
            if s.notna().any():
                thr = s.dropna().quantile(0.90)
                df["is_high_speech_artist"] = (s >= thr).astype(flag)

        if instr is not None:
            ins = pd.to_numeric(instr, errors="coerce")
            df["is_instrumental_heavy_artist"] = (ins >= 0.5).astype(flag)

        # loudness relative
        if loud is not None:
            l = pd.to_numeric(loud, errors="coerce")
            if l.notna().any():
                med = l.dropna().median()
                df["is_quiet_artist"] = (l < med).astype(flag)

        return df

    def _genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        flag = self.cfg.flag_dtype

        if "artist_genres" not in df.columns:
            return df

        genres = ensure_list_column(col_or_na(df, "artist_genres"))
        df["n_artist_genres"] = genres.apply(len).astype("int16")
        df["has_genre"] = (df["n_artist_genres"] > 0).astype(flag)
        df["main_genre_id"] = genres.apply(lambda xs: xs[0] if len(xs) else pd.NA).astype("string")
        df["is_multi_genre"] = (df["n_artist_genres"] >= 3).astype(flag)

        return df


