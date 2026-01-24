from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.catalog import TableCatalog
from ..features.time_features import col_or_na
from ..features.numeric_transform import log1p_numeric
from ..data.parsing import ensure_list_column

@dataclass(frozen=True)
class ArtistDatasetConfig:
    # welche Audiofeatures sollen in das Artist-Profil?
    policy_audio: tuple[str, ...] = (
        "acousticness", "danceability", "energy", "instrumentalness", "liveness",
        "speechiness", "valence", "loudness", "tempo"
    )
    add_track_popularity_agg: bool = True
    add_explicit_rate: bool = True


class ArtistDatasetBuilder:
    """
    Artist-level Dataset:
    artists + aggregated track profile (mean audio, n_tracks, optional popularity/explicit)
    + artist_genres as list
    """

    def __init__(self, config: ArtistDatasetConfig = ArtistDatasetConfig()):
        self.cfg = config

    def build(self, data: Dict[str, pd.DataFrame], *, track_df: pd.DataFrame) -> pd.DataFrame:
        artists = data.get("artists", pd.DataFrame()).copy()
        rta = data.get("r_track_artist", pd.DataFrame()).copy()
        rag = data.get("r_artist_genre", pd.DataFrame()).copy()

        # --- ids join-safe ---
        artists = TableCatalog.ensure_str_cols(artists, ["id"])
        rta = TableCatalog.ensure_str_cols(rta, ["track_id", "artist_id"])
        rag = TableCatalog.ensure_str_cols(rag, ["artist_id", "genre_id"])
        track_df = TableCatalog.ensure_str_cols(track_df.copy(), ["track_id"])

        # 1) base artist df
        artist_df = artists.rename(columns={"id": "artist_id"}).copy()

        # 2) artist -> genres (list)
        artist_df = self._join_artist_genres(artist_df, rag)

        # 3) aggregated artist style profile from track_df via rta
        if not rta.empty and not track_df.empty:
            artist_profile = self._aggregate_artist_profile(rta=rta, track_df=track_df)
            artist_df = artist_df.merge(artist_profile, on="artist_id", how="left")
        else:
            artist_df["n_tracks"] = pd.NA

        # 4) basic heavy-tail transforms
        artist_df["log_followers"] = log1p_numeric(col_or_na(artist_df, "followers"))
        artist_df["log_n_tracks"] = log1p_numeric(col_or_na(artist_df, "n_tracks"))

        return artist_df

    # ------------------------------------------------------------------

    def _join_artist_genres(self, artist_df: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
        if rag.empty:
            artist_df["artist_genres"] = [[] for _ in range(len(artist_df))]
            artist_df["artist_genres"] = ensure_list_column(col_or_na(artist_df, "artist_genres"))
            return artist_df

        artist_to_genres = (
            rag.groupby("artist_id")["genre_id"]
            .apply(lambda x: sorted(set(x.dropna().tolist())))
            .reset_index()
            .rename(columns={"genre_id": "artist_genres"})
        )

        out = artist_df.merge(artist_to_genres, on="artist_id", how="left")
        out["artist_genres"] = ensure_list_column(col_or_na(out, "artist_genres"))
        return out

    def _aggregate_artist_profile(self, *, rta: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame:
        # welche Track-Spalten brauchen wir?
        audio_cols_present: List[str] = [c for c in self.cfg.policy_audio if c in track_df.columns]

        cols = ["track_id"] + audio_cols_present

        if self.cfg.add_track_popularity_agg and "popularity" in track_df.columns:
            cols.append("popularity")

        if self.cfg.add_explicit_rate and "explicit" in track_df.columns:
            cols.append("explicit")

        # join rta (artist_id, track_id) + track features
        rta_feat = rta.merge(track_df[cols], on="track_id", how="left")

        def explicit_rate_fn(x: pd.Series) -> float:
            xx = pd.to_numeric(x, errors="coerce")
            if xx.dropna().empty:
                return np.nan
            return float(np.nanmean(xx))

        agg_dict = {
            "n_tracks": ("track_id", "nunique"),
        }

        if "popularity" in rta_feat.columns:
            agg_dict["track_pop_mean"] = ("popularity", "mean")

        if "explicit" in rta_feat.columns:
            agg_dict["explicit_rate"] = ("explicit", explicit_rate_fn)

        for c in audio_cols_present:
            agg_dict[f"mean_{c}"] = (c, "mean")

        artist_audio_agg = (
            rta_feat.groupby("artist_id")
            .agg(**agg_dict)
            .reset_index()
        )

        return artist_audio_agg