# utils/modeling/feature_schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class FeatureSchemaConfig:
    # Global policies
    allow_text_features: bool = True
    allow_reach_proxies: bool = False  # OFF by default (more realistic)
    allow_album_popularity_proxy: bool = False  # OFF by default (very leaky)

    # Which duration column to prefer if both exist
    prefer_duration_col: str = "duration_af"  # fallback "duration"

    # Audio policy (track-level)
    policy_audio: Tuple[str, ...] = (
        "acousticness", "danceability", "energy", "instrumentalness", "liveness",
        "speechiness", "valence", "loudness", "tempo"
    )

    policy_audio_extra: Tuple[str, ...] = ("key", "mode", "time_signature")

    # Core track columns you consider safe (if present)
    policy_track_base: Tuple[str, ...] = (
        "disc_number", "track_number",
        "duration", "duration_af", "log_duration",
        "has_preview", "has_audio_features",
        "n_artists", "is_collab",
        "release_year", "release_month", "release_decade",
        "is_modern_release", "is_old_release", "is_release_year_missing",
        "is_tracknum_extreme", "is_multidisc", "is_disc_extreme",
        "is_tracknum_missing", "is_disc_missing",
        "is_time_signature_rare", "is_tempo_extreme", "is_loudness_very_low",
        "is_af_long", "is_high_speech", "is_instrumental",
        "is_fast_tempo", "is_very_quiet",
        "mood_happy_energetic", "mood_angry_energetic", "mood_calm_happy", "mood_sad_calm",
        "is_long_track", "is_explicit",
    )

    # Reach proxies (strong leakage for popularity-derived targets)
    reach_proxy_cols: Tuple[str, ...] = (
        "artist_popularity_mean", "artist_popularity_max",
        "artist_followers_mean", "artist_followers_max",
        "log_artist_followers_mean", "log_artist_followers_max",
        "artist_followers_dispersion", "artist_popularity_dispersion",
        "album_popularity", "popularity_vs_album",
    )

    # Global NEVER columns (always excluded)
    never_cols: Set[str] = field(default_factory=lambda: {
        # IDs
        "track_id", "album_id", "artist_id", "audio_feature_id", "id",
        # URLs / URIs
        "analysis_url", "preview_url", "href", "uri", "spotify_url",
        # Raw list columns (use multi-hot instead)
        "artist_ids", "track_genres", "album_genres", "artist_genres",
        # Raw text
        "name", "name_album",
        # Raw dates
        "release_date", "release_date_parsed", "track_release_date_parsed", "album_release_date_raw",
        # Targets / leakage sources (extend per task)
        "popularity", "success_pct_in_cohort", "success_residual_in_cohort",
    })


# ----------------------------
# Builder / Selector
# ----------------------------
class FeatureSchemaBuilder:
    """
    Builds clean ML feature matrices from prepared tables.
    - Applies leakage guards
    - Selects features by policy groups
    - Optionally appends genre multi-hot matrices
    """

    def __init__(self, config: FeatureSchemaConfig = FeatureSchemaConfig()):
        self.cfg = config

    # ---------- public API ----------

    def build_track_matrix(
        self,
        track_df: pd.DataFrame,
        *,
        track_genre_mh: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        """
        Returns dict:
          - X_track (DataFrame)
          - schema (dict with columns used)
          - report (dict with dropped/missing)
        """
        df = track_df.copy()

        # 1) Collect candidates
        cols = []
        cols += self._present(df, list(self.cfg.policy_track_base))
        cols += self._present(df, list(self.cfg.policy_audio))
        cols += self._present(df, list(self.cfg.policy_audio_extra))

        # text engineered
        if self.cfg.allow_text_features:
            cols += self._present(df, ["name_len", "name_words"])

        # reach proxies
        if self.cfg.allow_reach_proxies:
            cols += self._present(df, list(self.cfg.reach_proxy_cols))

        # album popularity is extra risky (separate toggle)
        if not self.cfg.allow_album_popularity_proxy:
            cols = [c for c in cols if c != "album_popularity"]

        # 2) Resolve duplicate duration
        cols = self._resolve_duration(cols, df)

        # 3) Apply global NEVER policy
        cols_final, dropped_never = self._apply_never(cols)

        # 4) Build X
        X = df[cols_final].copy()

        # 5) Append genre multi-hot
        if track_genre_mh is not None and isinstance(track_genre_mh, pd.DataFrame) and track_genre_mh.shape[1] > 0:
            X = pd.concat([X.reset_index(drop=True), track_genre_mh.reset_index(drop=True)], axis=1)

        # 6) Report
        report = self._report(
            df=df,
            selected=cols_final,
            dropped_never=dropped_never,
            genre_cols=(list(track_genre_mh.columns) if isinstance(track_genre_mh, pd.DataFrame) else []),
        )

        return {
            "X_track": X,
            "schema": {"track_cols": cols_final},
            "report": report,
        }

    def build_album_matrix(
        self,
        album_df: pd.DataFrame,
        *,
        album_genre_mh: Optional[pd.DataFrame] = None,
        allow_popularity_proxy: bool = False,
    ) -> Dict[str, object]:
        df = album_df.copy()

        # Base safe columns (exclude raw name + ids by NEVER)
        base = [
            "album_type",
            "release_year", "release_month", "release_decade",
            "is_modern_release", "is_old_release", "is_release_year_missing",
            "release_year_norm",
            "n_tracks", "log_n_tracks",
            "is_single", "is_ep", "is_lp", "is_mega_album", "is_n_tracks_missing",
            "album_name_len", "album_name_words",
            "is_deluxe", "is_remaster", "is_live_album", "is_compilation",
            "album_dance_x_energy",
            "album_is_slow_tempo", "album_is_fast_tempo", "album_is_quiet",
            "album_is_high_speech", "album_is_instrumental_heavy",
            "album_is_collab",
            "album_artist_followers_mean_log1p", "album_artist_followers_max_log1p",
            "album_artist_followers_gap",
            "n_album_genres", "album_has_genre", "album_is_multi_genre",
        ]

        # album mean audio columns (dynamic)
        mean_audio = [c for c in df.columns if c.startswith("album_mean_")]

        cols = self._present(df, base) + mean_audio

        # Popularity proxies (OFF by default)
        if allow_popularity_proxy:
            cols += self._present(df, [
                "album_artist_popularity_mean", "album_artist_popularity_max",
                "album_artist_popularity_gap", "album_has_headliner_artist",
                "popularity",  # album popularity itself (very leaky depending on task)
            ])

        cols_final, dropped_never = self._apply_never(cols)

        X = df[cols_final].copy()
        if album_genre_mh is not None and isinstance(album_genre_mh, pd.DataFrame) and album_genre_mh.shape[1] > 0:
            X = pd.concat([X.reset_index(drop=True), album_genre_mh.reset_index(drop=True)], axis=1)

        report = self._report(
            df=df,
            selected=cols_final,
            dropped_never=dropped_never,
            genre_cols=(list(album_genre_mh.columns) if isinstance(album_genre_mh, pd.DataFrame) else []),
        )

        return {"X_album": X, "schema": {"album_cols": cols_final}, "report": report}

    def build_artist_matrix(
        self,
        artist_df: pd.DataFrame,
        *,
        artist_genre_mh: Optional[pd.DataFrame] = None,
        allow_popularity: bool = True,
    ) -> Dict[str, object]:
        df = artist_df.copy()

        base = [
            "followers", "followers_log1p", "log_followers",
            "n_tracks", "log_n_tracks", "n_tracks_log1p",
            "followers_per_track", "followers_per_track_log1p",
            "explicit_rate_filled", "is_mostly_explicit", "is_never_explicit",
            "dance_x_energy",
            "is_slow_tempo_artist", "is_fast_tempo_artist",
            "acoustic_minus_energy",
            "is_high_speech_artist", "is_instrumental_heavy_artist", "is_quiet_artist",
            "n_artist_genres", "has_genre", "is_multi_genre",
        ]

        mean_audio = [c for c in df.columns if c.startswith("mean_")]
        cols = self._present(df, base) + mean_audio

        if allow_popularity:
            cols += self._present(df, ["popularity", "is_headliner_popularity", "artist_vs_track_pop_gap"])

        cols_final, dropped_never = self._apply_never(cols)

        X = df[cols_final].copy()
        if artist_genre_mh is not None and isinstance(artist_genre_mh, pd.DataFrame) and artist_genre_mh.shape[1] > 0:
            X = pd.concat([X.reset_index(drop=True), artist_genre_mh.reset_index(drop=True)], axis=1)

        report = self._report(
            df=df,
            selected=cols_final,
            dropped_never=dropped_never,
            genre_cols=(list(artist_genre_mh.columns) if isinstance(artist_genre_mh, pd.DataFrame) else []),
        )

        return {"X_artist": X, "schema": {"artist_cols": cols_final}, "report": report}

    # ---------- internal helpers ----------

    @staticmethod
    def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    def _apply_never(self, cols: List[str]) -> Tuple[List[str], List[str]]:
        seen = set()
        dedup = []
        for c in cols:
            if c not in seen:
                dedup.append(c)
                seen.add(c)

        dropped = [c for c in dedup if c in self.cfg.never_cols]
        final = [c for c in dedup if c not in self.cfg.never_cols]
        return final, dropped

    def _resolve_duration(self, cols: List[str], df: pd.DataFrame) -> List[str]:
        # keep only one duration column (prefer cfg.prefer_duration_col if exists)
        duration_candidates = [c for c in ["duration_af", "duration"] if c in df.columns]
        if len(duration_candidates) <= 1:
            return cols

        prefer = self.cfg.prefer_duration_col
        keep = prefer if prefer in duration_candidates else duration_candidates[0]
        drop = [c for c in duration_candidates if c != keep]

        out = [c for c in cols if c not in drop]
        return out

    @staticmethod
    def _report(
        df: pd.DataFrame,
        selected: List[str],
        dropped_never: List[str],
        genre_cols: List[str],
    ) -> Dict[str, object]:
        missing = [c for c in selected if c not in df.columns]
        # (missing should be empty, but keep it for sanity)
        return {
            "n_rows": int(df.shape[0]),
            "n_selected": int(len(selected)),
            "selected_cols": selected,
            "dropped_never": dropped_never,
            "missing_after_select": missing,
            "n_genre_cols": int(len(genre_cols)),
            "genre_cols_preview": genre_cols[:10],
        }