# utils/datasets/track_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import pandas as pd


from ..features.time_features import col_or_na, add_release_time_features
from ..data.parsing import ensure_list_column
from ..data.parsing import parse_datetime_from_candidates
from ..data.catalog import TableCatalog


MainAlbumStrategy = Literal["earliest_release", "first"]


@dataclass(frozen=True)
class TrackDatasetConfig:
    """Konfiguration für den Track-Dataset-Aufbau."""
    main_album_strategy: MainAlbumStrategy = "earliest_release"


class TrackDatasetBuilder:
    """
    Baut ein Track-level Dataset:
    tracks + audio_features + main album + artists aggregations + genres + engineered features
    """

    def __init__(
        self,
        config: TrackDatasetConfig,
    ):
        self.config = config


    def build(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Einstiegspunkt: nimmt dein `data` Dict (aus Parquet) und liefert `track_df`.
        """
        # --- Tabellen holen ---
        tracks = data["tracks"].copy()
        audio = data.get("audio_features", pd.DataFrame()).copy()
        rat = data.get("r_albums_tracks", pd.DataFrame()).copy()
        albums = data.get("albums", pd.DataFrame()).copy()
        rta = data.get("r_track_artist", pd.DataFrame()).copy()
        rag = data.get("r_artist_genre", pd.DataFrame()).copy()
        artists = data.get("artists", pd.DataFrame()).copy()

        # --- ID Spalten join-sicher machen ---
        tracks = TableCatalog.ensure_str_cols(tracks, ["track_id", "audio_feature_id"])
        audio = TableCatalog.ensure_str_cols(audio, ["id"])
        rat = TableCatalog.ensure_str_cols(rat, ["album_id", "track_id"])
        albums = TableCatalog.ensure_str_cols(albums, ["id"])
        rta = TableCatalog.ensure_str_cols(rta, ["track_id", "artist_id"])
        rag = TableCatalog.ensure_str_cols(rag, ["artist_id", "genre_id"])
        artists = TableCatalog.ensure_str_cols(artists, ["id"])

        # --- 1) tracks -> audio_features (LEFT JOIN) ---
        track_df = self._join_audio_features(tracks, audio)

        # --- 2) track -> main album auswählen ---
        main_album = self._select_main_album_per_track(rat, albums)
        track_df = track_df.merge(main_album, on="track_id", how="left")

        # --- 3) album metadata join ---
        track_df = self._join_album_metadata(track_df, albums)

        # --- 5) artist aggregations ---
        artist_agg = self._aggregate_artists_per_track(rta, artists)
        track_df = track_df.merge(artist_agg, on="track_id", how="left")

        # --- 6) genres via artist genres ---
        track_genres = self._aggregate_genres_per_track(rta, rag)
        track_df = track_df.merge(track_genres, on="track_id", how="left")
        track_df["track_genres"] = ensure_list_column(col_or_na(track_df, "track_genres"))

        return track_df

    # ------------------------
    # Teil-Schritte
    # ------------------------

    def _join_audio_features(self, tracks: pd.DataFrame, audio: pd.DataFrame) -> pd.DataFrame:
        # tracks muss audio_feature_id haben, sonst kannst du nicht joinen
        if "audio_feature_id" not in tracks.columns:
            raise ValueError("tracks muss audio_feature_id enthalten, um audio_features zu joinen.")

        # join keys als string (wichtig, sonst merge fail wegen dtype mismatch)
        tracks = TableCatalog.ensure_str_cols(tracks.copy(), ["track_id", "audio_feature_id"])
        audio = TableCatalog.ensure_str_cols(audio.copy(), ["id"])

        # wenn audio leer: einfach tracks zurück
        if audio.empty:
            out = tracks
        else:
            # audio id -> audio_feature_id für den Join
            audio_small = audio.rename(columns={"id": "audio_feature_id"}).copy()

            # Optional: falls audio_small selbst eine "audio_feature_id" doppelt enthält
            audio_small = audio_small.drop_duplicates(subset=["audio_feature_id"])

            # Merge
            out = tracks.merge(audio_small, on="audio_feature_id", how="left", suffixes=("", "_af"))

        # optional: track release date fallback (falls vorhanden)
        out = parse_datetime_from_candidates(
            out,
            candidates=["release_date", "track_release_date", "track_release_date_parsed"],
            out_col="track_release_date_parsed",
        )
        return out

    def _select_main_album_per_track(self, rat: pd.DataFrame, albums: pd.DataFrame) -> pd.DataFrame:
        if rat.empty or albums.empty:
            return pd.DataFrame({"track_id": rat.get("track_id", pd.Series(dtype="string")), "album_id": pd.NA}).drop_duplicates()

        albums2 = albums.copy()
        if "album_id" not in albums2.columns and "id" in albums2.columns:
            albums2 = albums2.rename(columns={"id": "album_id"})

        albums2 = parse_datetime_from_candidates(
            albums2,
            candidates=["release_date_parsed", "release_date", "album_release_date_parsed", "album_release_date"],
            out_col="album_release_date_parsed",
        )

        rat2 = rat.merge(albums2[["album_id", "album_release_date_parsed"]], on="album_id", how="left")

        if self.config.main_album_strategy == "earliest_release":
            rat2 = rat2.sort_values(["track_id", "album_release_date_parsed", "album_id"], ascending=[True, True, True])
        else:
            rat2 = rat2.sort_values(["track_id", "album_id"], ascending=[True, True])

        return rat2.drop_duplicates("track_id", keep="first")[["track_id", "album_id"]]

    def _join_album_metadata(self, track_df: pd.DataFrame, albums: pd.DataFrame) -> pd.DataFrame:
        if albums.empty or track_df.empty or "album_id" not in track_df.columns:
            return track_df

        out = track_df.copy()

        # ensure the final date comes ONLY from album (avoid merge suffix confusion)
        if "release_date_parsed" in out.columns:
            out = out.drop(columns=["release_date_parsed"])

        albums_join = albums.copy()

        # ensure join key
        if "album_id" not in albums_join.columns:
            if "id" in albums_join.columns:
                albums_join = albums_join.rename(columns={"id": "album_id"})
            else:
                return out

        # optional collision protection for other columns
        if "popularity" in albums_join.columns:
            albums_join = albums_join.rename(columns={"popularity": "album_popularity"})
        if "release_date" in albums_join.columns:
            albums_join = albums_join.rename(columns={"release_date": "album_release_date_raw"})

        # ---- build release_date_parsed (album truth)
        if "release_date_parsed" in albums_join.columns:
            albums_join["release_date_parsed"] = pd.to_datetime(albums_join["release_date_parsed"], errors="coerce")
        else:
            albums_join["release_date_parsed"] = pd.NaT

        # keep only needed album columns
        keep_cols = ["album_id", "release_date_parsed"]
        if "album_popularity" in albums_join.columns:
            keep_cols.append("album_popularity")
        if "album_release_date_raw" in albums_join.columns:
            keep_cols.append("album_release_date_raw")

        albums_join = albums_join[keep_cols].drop_duplicates("album_id")

        # merge: each track gets the date from its album_id
        out = out.merge(albums_join, on="album_id", how="left")

        # optional time features
        out = add_release_time_features(out, "release_date_parsed")

        # optional: remove tracks whose album has no date
        # out = out[out["release_date_parsed"].notna()]

        return out

    def _aggregate_artists_per_track(self, rta: pd.DataFrame, artists: pd.DataFrame) -> pd.DataFrame:
        if rta.empty or artists.empty:
            return pd.DataFrame({"track_id": pd.Series(dtype="string")})

        artist_feat = artists.rename(columns={"id": "artist_id", "popularity": "artist_popularity", "followers": "artist_followers"}).copy()
        rta_art = rta.merge(artist_feat, on="artist_id", how="left")

        agg = (
            rta_art.groupby("track_id")
            .agg(
                artist_ids=("artist_id", lambda x: sorted(set(x.dropna().tolist()))),
                n_artists=("artist_id", "nunique"),
                artist_popularity_mean=("artist_popularity", "mean"),
                artist_popularity_max=("artist_popularity", "max"),
                artist_followers_mean=("artist_followers", "mean"),
                artist_followers_max=("artist_followers", "max"),
            )
            .reset_index()
        )
        return agg

    def _aggregate_genres_per_track(self, rta: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
        if rta.empty or rag.empty:
            return pd.DataFrame({"track_id": pd.Series(dtype="string"), "track_genres": [[]]})

        artist_to_genres = (
            rag.groupby("artist_id")["genre_id"]
            .apply(lambda x: sorted(set(x.dropna().tolist())))
            .reset_index()
            .rename(columns={"genre_id": "artist_genres"})
        )

        rta_gen = rta.merge(artist_to_genres, on="artist_id", how="left")

        track_to_genres = (
            rta_gen.groupby("track_id")["artist_genres"]
            .apply(lambda rows: sorted(set([g for lst in rows.dropna() for g in (lst if isinstance(lst, list) else [])])))
            .reset_index()
            .rename(columns={"artist_genres": "track_genres"})
        )
        return track_to_genres