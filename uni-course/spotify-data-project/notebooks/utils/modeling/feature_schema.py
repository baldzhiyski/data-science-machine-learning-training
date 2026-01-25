# utils/modeling/feature_schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Literal

import pandas as pd


TaskName = Literal[
    "success_pct",
    "success_residual",
    "hit",
    "mood",
    "ranker",
    "generic",
]


@dataclass(frozen=True)
class FeatureSchemaConfig:
    allow_text_features: bool = True
    allow_reach_proxies: bool = False
    allow_album_popularity_proxy: bool = False
    prefer_duration_col: str = "duration_af"

    policy_audio: Tuple[str, ...] = (
        "acousticness", "danceability", "energy", "instrumentalness", "liveness",
        "speechiness", "valence", "loudness", "tempo"
    )
    policy_audio_extra: Tuple[str, ...] = ("key", "mode", "time_signature")

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

    reach_proxy_cols: Tuple[str, ...] = (
        "artist_popularity_mean", "artist_popularity_max",
        "artist_followers_mean", "artist_followers_max",
        "log_artist_followers_mean", "log_artist_followers_max",
        "artist_followers_dispersion", "artist_popularity_dispersion",
        "album_popularity", "popularity_vs_album",
    )

    # Global NEVER (always excluded)
    never_cols: Set[str] = field(default_factory=lambda: {
        "track_id", "album_id", "artist_id", "audio_feature_id", "id",
        "analysis_url", "preview_url", "href", "uri", "spotify_url",
        "artist_ids", "track_genres", "album_genres", "artist_genres",
        "name", "name_album",
        "release_date", "release_date_parsed", "track_release_date_parsed", "album_release_date_raw",
        "popularity", "success_pct_in_cohort", "success_residual_in_cohort",
    })

    # Task-specific leakage extras
    task_never: Dict[str, Set[str]] = field(default_factory=lambda: {
        # popularity-derived targets → exclude all popularity proxies by default
        "success_pct": {"album_popularity", "artist_popularity_mean", "artist_popularity_max"},
        "success_residual": {"album_popularity", "artist_popularity_mean", "artist_popularity_max"},
        "hit": {"album_popularity", "artist_popularity_mean", "artist_popularity_max"},
        # mood can optionally allow more metadata (still safe to keep strict)
        "mood": set(),
        "ranker": set(),
        "generic": set(),
    })


class FeatureSchemaBuilder:
    def __init__(self, config: FeatureSchemaConfig = FeatureSchemaConfig()):
        self.cfg = config

    # ---------------------------
    # Track
    # ---------------------------
    def build_track_matrix(
        self,
        track_df: pd.DataFrame,
        *,
        task: TaskName = "generic",
        track_genre_mh: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        df = track_df.copy()

        groups: Dict[str, List[str]] = {}

        groups["base"] = self._present(df, list(self.cfg.policy_track_base))
        groups["audio"] = self._present(df, list(self.cfg.policy_audio))
        groups["audio_extra"] = self._present(df, list(self.cfg.policy_audio_extra))

        if self.cfg.allow_text_features:
            groups["text"] = self._present(df, ["name_len", "name_words"])
        else:
            groups["text"] = []

        if self.cfg.allow_reach_proxies:
            groups["reach"] = self._present(df, list(self.cfg.reach_proxy_cols))
        else:
            groups["reach"] = []

        # album popularity special toggle
        if not self.cfg.allow_album_popularity_proxy:
            groups["reach"] = [c for c in groups["reach"] if c != "album_popularity"]

        # flatten
        cols = self._dedup(sum(groups.values(), []))

        # resolve duration duplicates
        cols = self._resolve_duration(cols, df)

        # apply never + task never
        cols_final, dropped_never = self._apply_never(cols, task=task)

        X = df[cols_final].copy()

        # Append genre multi-hot (align safely)
        genre_cols = []
        if isinstance(track_genre_mh, pd.DataFrame) and track_genre_mh.shape[1] > 0:
            # if indices match, align by index; else fallback to row order
            if track_genre_mh.index.equals(df.index):
                X = pd.concat([X, track_genre_mh], axis=1)
            else:
                X = pd.concat([X.reset_index(drop=True), track_genre_mh.reset_index(drop=True)], axis=1)
            genre_cols = list(track_genre_mh.columns)

        report = self._report(df, cols_final, dropped_never, genre_cols, groups, task)

        return {"X_track": X, "schema": {"track_cols": cols_final}, "report": report}

    # ---------------------------
    # Album
    # ---------------------------
    def build_album_matrix(
        self,
        album_df: pd.DataFrame,
        *,
        album_genre_mh: Optional[pd.DataFrame] = None,
        allow_popularity_proxy: bool = False,
    ) -> Dict[str, object]:
        df = album_df.copy()

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

        mean_audio = [c for c in df.columns if c.startswith("album_mean_")]
        cols = self._present(df, base) + mean_audio

        if allow_popularity_proxy:
            cols += self._present(df, [
                "album_artist_popularity_mean", "album_artist_popularity_max",
                "album_artist_popularity_gap", "album_has_headliner_artist",
                "popularity",
            ])

        cols = self._dedup(cols)
        cols_final, dropped_never = self._apply_never(cols, task="generic")

        X = df[cols_final].copy()

        genre_cols = []
        if isinstance(album_genre_mh, pd.DataFrame) and album_genre_mh.shape[1] > 0:
            if album_genre_mh.index.equals(df.index):
                X = pd.concat([X, album_genre_mh], axis=1)
            else:
                X = pd.concat([X.reset_index(drop=True), album_genre_mh.reset_index(drop=True)], axis=1)
            genre_cols = list(album_genre_mh.columns)

        report = self._report(df, cols_final, dropped_never, genre_cols, groups=None, task="generic")
        return {"X_album": X, "schema": {"album_cols": cols_final}, "report": report}

    # ---------------------------
    # Artist (same idea as yours, shortened here)
    # ---------------------------
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

        cols = self._dedup(cols)
        cols_final, dropped_never = self._apply_never(cols, task="generic")

        X = df[cols_final].copy()

        genre_cols = []
        if isinstance(artist_genre_mh, pd.DataFrame) and artist_genre_mh.shape[1] > 0:
            if artist_genre_mh.index.equals(df.index):
                X = pd.concat([X, artist_genre_mh], axis=1)
            else:
                X = pd.concat([X.reset_index(drop=True), artist_genre_mh.reset_index(drop=True)], axis=1)
            genre_cols = list(artist_genre_mh.columns)

        report = self._report(df, cols_final, dropped_never, genre_cols, groups=None, task="generic")
        return {"X_artist": X, "schema": {"artist_cols": cols_final}, "report": report}

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _dedup(cols: List[str]) -> List[str]:
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _apply_never(self, cols: List[str], *, task: TaskName) -> Tuple[List[str], List[str]]:
        task_block = self.cfg.task_never.get(task, set())
        blocked = self.cfg.never_cols.union(task_block)
        dropped = [c for c in cols if c in blocked]
        final = [c for c in cols if c not in blocked]
        return final, dropped

    def _resolve_duration(self, cols: List[str], df: pd.DataFrame) -> List[str]:
        duration_candidates = [c for c in ["duration_af", "duration"] if c in df.columns and c in cols]
        if len(duration_candidates) <= 1:
            return cols

        prefer = self.cfg.prefer_duration_col
        keep = prefer if prefer in duration_candidates else duration_candidates[0]
        drop = [c for c in duration_candidates if c != keep]
        return [c for c in cols if c not in drop]

    @staticmethod
    def _report(
        df: pd.DataFrame,
        selected: List[str],
        dropped_never: List[str],
        genre_cols: List[str],
        groups: Optional[Dict[str, List[str]]],
        task: str,
    ) -> Dict[str, object]:
        base = {
            "task": task,
            "n_rows": int(df.shape[0]),
            "n_selected": int(len(selected)),
            "selected_preview": selected[:25],
            "dropped_never": dropped_never,
            "n_genre_cols": int(len(genre_cols)),
            "genre_preview": genre_cols[:10],
        }
        if groups:
            base["group_counts"] = {k: len(v) for k, v in groups.items()}
        return base