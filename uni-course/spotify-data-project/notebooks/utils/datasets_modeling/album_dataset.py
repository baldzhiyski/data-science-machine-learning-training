# utils/datasets_modeling/album_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..data.catalog import TableCatalog
from ..features.time_features import col_or_na, add_release_time_features
from ..features.numeric_transform import log1p_numeric
import importlib
from ..data import parsing

importlib.reload(parsing)
from ..data.parsing import ensure_list_column, parse_datetime_from_candidates,safe_len_series, safe_word_count_series


@dataclass(frozen=True)
class AlbumDatasetConfig:
    policy_audio: tuple[str, ...] = (
        "acousticness", "danceability", "energy", "instrumentalness", "liveness",
        "speechiness", "valence", "loudness", "tempo"
    )


class AlbumDatasetBuilder:
    """
    Album-level Dataset:
    albums + n_tracks + album_mean_audio + artist aggs + album_genres + basic features
    (Komplett aus raw `data` Tabellen — kein track_df nötig)
    """

    def __init__(self, config: AlbumDatasetConfig = AlbumDatasetConfig()):
        self.cfg = config

    def build(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        albums = data.get("albums", pd.DataFrame()).copy()
        tracks = data.get("tracks", pd.DataFrame()).copy()
        audio = data.get("audio_features", pd.DataFrame()).copy()

        rat = data.get("r_albums_tracks", pd.DataFrame()).copy()
        raa = data.get("r_albums_artists", pd.DataFrame()).copy()
        rag = data.get("r_artist_genre", pd.DataFrame()).copy()
        artists = data.get("artists", pd.DataFrame()).copy()

        # --- join keys string-safe ---
        albums = TableCatalog.ensure_str_cols(albums, ["id"])
        tracks = TableCatalog.ensure_str_cols(tracks, ["track_id", "audio_feature_id"])
        audio = TableCatalog.ensure_str_cols(audio, ["id"])
        rat = TableCatalog.ensure_str_cols(rat, ["album_id", "track_id"])
        raa = TableCatalog.ensure_str_cols(raa, ["album_id", "artist_id"])
        rag = TableCatalog.ensure_str_cols(rag, ["artist_id", "genre_id"])
        artists = TableCatalog.ensure_str_cols(artists, ["id"])

        # 1) base album df
        album_df = albums.rename(columns={"id": "album_id"}).copy()

        album_df = parse_datetime_from_candidates(
            album_df,
            candidates=["release_date_parsed", "release_date", "album_release_date_parsed", "album_release_date"],
            out_col="release_date_parsed",
        )
        album_df = add_release_time_features(album_df, "release_date_parsed")

        # 2) n_tracks per album
        if not rat.empty:
            album_track_counts = (
                rat.groupby("album_id")["track_id"]
                .nunique()
                .reset_index()
                .rename(columns={"track_id": "n_tracks"})
            )
            album_df = album_df.merge(album_track_counts, on="album_id", how="left")
        else:
            album_df["n_tracks"] = pd.NA

        # 3) build minimal track_audio_df from raw tracks+audio_features
        track_audio_df = self._build_track_audio_df(tracks, audio)

        # album mean audio
        audio_cols_present: List[str] = [c for c in self.cfg.policy_audio if c in track_audio_df.columns]
        if audio_cols_present and not rat.empty and not track_audio_df.empty:
            rat_track_audio = rat.merge(
                track_audio_df[["track_id"] + audio_cols_present],
                on="track_id",
                how="left",
            )

            album_audio_agg = (
                rat_track_audio.groupby("album_id")[audio_cols_present]
                .mean()
                .reset_index()
            )
            album_audio_agg = album_audio_agg.add_prefix("album_mean_").rename(
                columns={"album_mean_album_id": "album_id"}
            )
            album_df = album_df.merge(album_audio_agg, on="album_id", how="left")

        # 4) album artist aggs
        if not raa.empty and not artists.empty:
            artist_feat = artists.rename(
                columns={"id": "artist_id", "popularity": "artist_popularity", "followers": "artist_followers"}
            ).copy()

            raa_art = raa.merge(artist_feat, on="artist_id", how="left")
            album_artist_agg = (
                raa_art.groupby("album_id")
                .agg(
                    n_album_artists=("artist_id", "nunique"),
                    album_artist_popularity_mean=("artist_popularity", "mean"),
                    album_artist_popularity_max=("artist_popularity", "max"),
                    album_artist_followers_mean=("artist_followers", "mean"),
                    album_artist_followers_max=("artist_followers", "max"),
                )
                .reset_index()
            )
            album_df = album_df.merge(album_artist_agg, on="album_id", how="left")

        # 5) album genres
        if not raa.empty and not rag.empty:
            artist_to_genres = (
                rag.groupby("artist_id")["genre_id"]
                .apply(lambda x: sorted(set(x.dropna().tolist())))
                .reset_index()
                .rename(columns={"genre_id": "artist_genres"})
            )

            raa_gen = raa.merge(artist_to_genres, on="artist_id", how="left")

            album_to_genres = (
                raa_gen.groupby("album_id")["artist_genres"]
                .apply(lambda rows: sorted(set(
                    g for lst in rows.dropna()
                    for g in (lst if isinstance(lst, list) else [])
                )))
                .reset_index()
                .rename(columns={"artist_genres": "album_genres"})
            )

            album_df = album_df.merge(album_to_genres, on="album_id", how="left")
        else:
            album_df["album_genres"] = [[] for _ in range(len(album_df))]

        album_df["album_genres"] = ensure_list_column(col_or_na(album_df, "album_genres"))

        # 6) basic features
        album_df["log_n_tracks"] = log1p_numeric(col_or_na(album_df, "n_tracks"))
        album_df["name_len"] = safe_len_series(col_or_na(album_df, "name"))
        album_df["name_words"] = safe_word_count_series(col_or_na(album_df, "name"))

        return album_df

    def _build_track_audio_df(self, tracks: pd.DataFrame, audio: pd.DataFrame) -> pd.DataFrame:
        if tracks.empty or audio.empty or "audio_feature_id" not in tracks.columns:
            return tracks[["track_id"]].copy() if "track_id" in tracks.columns else pd.DataFrame()

        audio_small = audio.rename(columns={"id": "audio_feature_id"}).drop_duplicates(subset=["audio_feature_id"])
        out = tracks.merge(audio_small, on="audio_feature_id", how="left")

        # keep only track_id + audio policy columns that exist
        keep = ["track_id"] + [c for c in self.cfg.policy_audio if c in out.columns]
        return out[keep].copy()