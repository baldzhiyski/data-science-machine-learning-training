"""
Unified Music Data Analysis Framework
Enthält alle Analyzer-Klassen für Spotify-Datenanalyse.
"""


import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseAnalyzer(ABC):
    """
    Base class for analyzers: shared IO, column utilities, plotting helpers.
    """

    def __init__(
        self,
        data_dir: Path,
        schema_reports_dir: Path,
        output_subdir: str,
        log_level: int = logging.INFO,
        logger: Optional[logging.Logger] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(schema_reports_dir) / output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    @staticmethod
    def pick_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
        """Pick first existing column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")
        return None

    @staticmethod
    def identify_audio_features(df: pd.DataFrame) -> List[str]:
        """Identify available audio features."""
        candidates = ["tempo", "energy", "valence", "loudness", "danceability"]
        return [c for c in candidates if c in df.columns]

    def save_csv(self, df: pd.DataFrame, filename: str, index: bool = True, encoding: str = "utf-8") -> Path:
        path = self.output_dir / filename
        df.to_csv(path, encoding=encoding, index=index)
        return path

    def save_plot(self, filename: str, dpi: int = 150) -> Path:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        return path

    @abstractmethod
    def execute(self, **kwargs):
        pass


# ============================================================================
# TREND ANALYZER
# ============================================================================

class TrendAnalyzer(BaseAnalyzer):
    """
    Analyze time trends of audio features by release year.
    """

    def __init__(self, data_dir: Path, schema_reports_dir: Path, log_level: int = logging.INFO):
        super().__init__(data_dir, schema_reports_dir, "trends", log_level=log_level)
        self.df: Optional[pd.DataFrame] = None
        self.yearly: Optional[pd.DataFrame] = None

    def load_and_merge_data(self) -> pd.DataFrame:
        tracks = pd.read_csv(self.data_dir / "tracks.csv").rename(columns={"id": "track_id"})
        audio = pd.read_csv(self.data_dir / "audio_features.csv").rename(columns={"id": "audio_feature_id"})

        albums = pd.read_csv(self.data_dir / "albums.csv")
        album_id_col = self.pick_column(albums, ["id", "album_id"])
        release_date_col = self.pick_column(albums, ["release_date"], required=False)

        albums_subset = albums[[album_id_col] + ([release_date_col] if release_date_col else [])].rename(
            columns={album_id_col: "album_id"}
        )

        df = tracks.merge(audio, on="audio_feature_id", how="left")

        # Attach album_id via relation table if available, else fallback to tracks.album_id if present
        rat_path = self.data_dir / "r_albums_tracks.csv"
        if rat_path.exists():
            r_at = pd.read_csv(rat_path)
            if "track_id" not in r_at.columns:
                # try common alternatives
                r_at_track = self.pick_column(r_at, ["track_id", "trackId", "trackID"])
                r_at_album = self.pick_column(r_at, ["album_id", "albumId", "albumID"])
                r_at = r_at.rename(columns={r_at_track: "track_id", r_at_album: "album_id"})
            df = df.merge(r_at[["track_id", "album_id"]], on="track_id", how="left")
        else:
            if "album_id" not in df.columns:
                df["album_id"] = pd.NA

        df = df.merge(albums_subset, on="album_id", how="left")

        # Parse release_date (ms timestamps expected)
        if release_date_col and release_date_col in df.columns:
            timestamps = pd.to_numeric(df[release_date_col], errors="coerce")
            df["release_dt"] = pd.to_datetime(timestamps, unit="ms", errors="coerce")
            df["year"] = df["release_dt"].dt.year
        else:
            df["release_dt"] = pd.NaT
            df["year"] = pd.NA

        return df

    def calculate_yearly_trends(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        min_year_count: int = 50,
    ) -> pd.DataFrame:
        yearly = (
            df.dropna(subset=["year"])
            .groupby("year")[feature_cols]
            .mean()
            .round(3)
        )
        yearly["count"] = df.groupby("year")["track_id"].count()
        return yearly[yearly["count"] >= min_year_count]

    def plot_trend(self, series: pd.Series, feature_name: str, rolling_window: int = 3) -> None:
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=series.index, y=series.values, marker="o", label="year")

        if len(series) >= 1 and rolling_window >= 2:
            rolling = series.rolling(rolling_window, min_periods=1).mean()
            sns.lineplot(x=rolling.index, y=rolling.values, linestyle="--", label=f"rolling_{rolling_window}")

        plt.title(f"Trend: {feature_name}")
        plt.xlabel("year")
        plt.ylabel(feature_name)
        plt.legend()

        self.save_plot(f"trend_{feature_name}.png", dpi=150)

    def execute(
        self,
        min_year_count: int = 50,
        rolling_window: int = 3,
        save_yearly_csv: bool = True,
    ) -> pd.DataFrame:
        self.logger.info("Running TrendAnalyzer")

        self.df = self.load_and_merge_data()
        feature_cols = self.identify_audio_features(self.df)

        self.logger.info("Detected features: %s", ", ".join(feature_cols) if feature_cols else "none")

        self.yearly = self.calculate_yearly_trends(self.df, feature_cols, min_year_count=min_year_count)

        if save_yearly_csv:
            path = self.save_csv(self.yearly, "yearly_trends.csv", index=True, encoding="utf-8")
            self.logger.info("Saved: %s", path)

        for col in feature_cols:
            series = self.yearly[col].dropna()
            if len(series) > 0:
                self.plot_trend(series, col, rolling_window=rolling_window)

        self.logger.info("Outputs in: %s", self.output_dir.resolve())
        return self.yearly


# ============================================================================
# CATEGORY ANALYZER
# ============================================================================

class CategoryAnalyzer(BaseAnalyzer):
    """
    Analyze audio features by categorical variables (album_type, genre_id).
    """

    def __init__(self, data_dir: Path, schema_reports_dir: Path, log_level: int = logging.INFO):
        super().__init__(data_dir, schema_reports_dir, "categories", log_level=log_level)
        self.df: Optional[pd.DataFrame] = None

    def load_and_merge_base_data(self) -> pd.DataFrame:
        tracks = pd.read_csv(self.data_dir / "tracks.csv").rename(columns={"id": "track_id"})
        audio = pd.read_csv(self.data_dir / "audio_features.csv").rename(columns={"id": "audio_feature_id"})
        albums = pd.read_csv(self.data_dir / "albums.csv")

        album_id_col = self.pick_column(albums, ["id", "album_id"])
        album_type_col = self.pick_column(albums, ["album_type"], required=False)

        albums_cols = [album_id_col]
        if album_type_col:
            albums_cols.append(album_type_col)

        albums_subset = albums[albums_cols].rename(columns={album_id_col: "album_id"})
        if album_type_col and album_type_col != "album_type":
            albums_subset = albums_subset.rename(columns={album_type_col: "album_type"})

        df = tracks.merge(audio, on="audio_feature_id", how="left")

        # Attach album_id via relation table if available
        rat_path = self.data_dir / "r_albums_tracks.csv"
        if rat_path.exists():
            r_at = pd.read_csv(rat_path)
            if "track_id" not in r_at.columns or "album_id" not in r_at.columns:
                r_at_track = self.pick_column(r_at, ["track_id", "trackId", "trackID"])
                r_at_album = self.pick_column(r_at, ["album_id", "albumId", "albumID"])
                r_at = r_at.rename(columns={r_at_track: "track_id", r_at_album: "album_id"})
            df = df.merge(r_at[["track_id", "album_id"]], on="track_id", how="left")
        else:
            if "album_id" not in df.columns:
                df["album_id"] = pd.NA

        df = df.merge(albums_subset, on="album_id", how="left")
        return df

    def create_boxplots(self, df: pd.DataFrame, category_col: str, numeric_cols: List[str]) -> None:
        """Create boxplots and only save them (no plt.show())."""
        for y_col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=df, x=category_col, y=y_col)
            plt.title(f"{y_col} by {category_col}")
            plt.xticks(rotation=30, ha="right")

            filename = f"box_{y_col}_by_{category_col}.png"
            self.save_plot(filename, dpi=150)

    def calculate_group_statistics(self, df: pd.DataFrame, category_col: str, numeric_cols: List[str]) -> pd.DataFrame:
        return (
            df.groupby(category_col)[numeric_cols]
            .agg(["mean", "std", "median", "count"])
            .round(2)
        )

    def analyze_by_album_type(self, save_stats_csv: bool = True) -> pd.DataFrame:
        self.df = self.load_and_merge_base_data()
        num_cols = self.identify_audio_features(self.df)

        if "album_type" not in self.df.columns:
            self.logger.warning("album_type not found. Returning empty stats.")
            return pd.DataFrame()

        self.logger.info("Album-type analysis with features: %s", ", ".join(num_cols) if num_cols else "none")

        self.create_boxplots(self.df.dropna(subset=["album_type"]), "album_type", num_cols)
        group_stats = self.calculate_group_statistics(self.df.dropna(subset=["album_type"]), "album_type", num_cols)

        if save_stats_csv:
            path = self.save_csv(group_stats, "stats_album_type.csv", encoding="utf-8", index=True)
            self.logger.info("Saved: %s", path)

        return group_stats

    def map_tracks_to_genres(self) -> pd.DataFrame:
        """
        Map tracks to genres via:
          r_track_artist (track -> artist) and r_artist_genre (artist -> genre_id)

        Note: Uses genre_id as the category (no genre_name).
        """
        genres = pd.read_csv(self.data_dir / "genres.csv")
        rag = pd.read_csv(self.data_dir / "r_artist_genre.csv")
        rta = pd.read_csv(self.data_dir / "r_track_artist.csv")

        # normalize expected columns (robust to small naming variations)
        rag_artist = self.pick_column(rag, ["artist_id", "artistId", "artistID"])
        rag_genre = self.pick_column(rag, ["genre_id", "genreId", "genreID"])

        rta_track = self.pick_column(rta, ["track_id", "trackId", "trackID"])
        rta_artist = self.pick_column(rta, ["artist_id", "artistId", "artistID"])

        rag_norm = rag[[rag_artist, rag_genre]].rename(columns={rag_artist: "artist_id", rag_genre: "genre_id"})
        rta_norm = rta[[rta_track, rta_artist]].rename(columns={rta_track: "track_id", rta_artist: "artist_id"})

        # pick one genre per artist (first occurrence)
        artist_genre = rag_norm.dropna(subset=["genre_id"]).drop_duplicates(subset=["artist_id"])

        # Track -> Artist -> Genre
        track_genre = rta_norm.merge(artist_genre, on="artist_id", how="left")[["track_id", "genre_id"]]

        # Base data (track/audio features)
        if self.df is None:
            self.df = self.load_and_merge_base_data()

        df_genre = self.df[["track_id", "audio_feature_id"]].merge(track_genre, on="track_id", how="left")

        # Merge audio features (numeric columns)
        audio = pd.read_csv(self.data_dir / "audio_features.csv").rename(columns={"id": "audio_feature_id"})
        df_genre = df_genre.merge(audio, on="audio_feature_id", how="left")

        # optional: keep only valid genre_ids that exist in genres.csv (if schema is consistent)
        if "id" in genres.columns:
            df_genre = df_genre[df_genre["genre_id"].isin(set(genres["id"])) | df_genre["genre_id"].isna()]

        return df_genre

    def analyze_by_genre(self, top_n: int = 15, save_stats_csv: bool = True) -> Optional[pd.DataFrame]:
        required_files = ["genres.csv", "r_artist_genre.csv", "r_track_artist.csv"]
        if not all((self.data_dir / f).exists() for f in required_files):
            self.logger.warning("Genre files missing. Skipping genre analysis.")
            return None

        self.logger.info("Genre analysis (top_n=%d)", top_n)

        df_genre = self.map_tracks_to_genres()
        if "genre_id" not in df_genre.columns:
            self.logger.warning("genre_id not available after mapping. Skipping.")
            return None

        # Top N by genre_id
        top_genres = df_genre["genre_id"].value_counts().head(top_n).index
        df_filtered = df_genre[df_genre["genre_id"].isin(top_genres)].copy()

        # For plotting, treat as categorical
        df_filtered["genre_id"] = df_filtered["genre_id"].astype(str)

        num_cols = self.identify_audio_features(df_filtered)

        self.create_boxplots(df_filtered.dropna(subset=["genre_id"]), "genre_id", num_cols)
        genre_stats = self.calculate_group_statistics(df_filtered.dropna(subset=["genre_id"]), "genre_id", num_cols)

        if save_stats_csv:
            path = self.save_csv(genre_stats, f"stats_genre_top{top_n}.csv", encoding="utf-8", index=True)
            self.logger.info("Saved: %s", path)

        return genre_stats

    def execute(self, include_genre: bool = True, top_genres: int = 15) -> Dict[str, Optional[pd.DataFrame]]:
        self.logger.info("Running CategoryAnalyzer")

        results: Dict[str, Optional[pd.DataFrame]] = {}
        results["album_type"] = self.analyze_by_album_type(save_stats_csv=True)

        if include_genre:
            results["genre"] = self.analyze_by_genre(top_n=top_genres, save_stats_csv=True)
        else:
            results["genre"] = None

        self.logger.info("Outputs in: %s", self.output_dir.resolve())
        return results


# ============================================================================
# INFLUENCE ANALYZER
# ============================================================================

class InfluenceAnalyzer(BaseAnalyzer):
    """
    Analyze influence of artist followers on average track popularity.
    """

    def __init__(self, data_dir: Path, schema_reports_dir: Path, log_level: int = logging.INFO):
        super().__init__(data_dir, schema_reports_dir, "influence", log_level=log_level)
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, Dict[str, str]] = {}

    def load_and_normalize_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tracks = pd.read_csv(self.data_dir / "tracks.csv", encoding="utf-8")
        artists = pd.read_csv(self.data_dir / "artists.csv", encoding="utf-8")
        rta = pd.read_csv(self.data_dir / "r_track_artist.csv", encoding="utf-8")

        track_id_col = self.pick_column(tracks, ["track_id", "id"])
        track_pop_col = self.pick_column(tracks, ["popularity"])

        artist_id_col = self.pick_column(artists, ["artist_id", "id"])
        followers_col = self.pick_column(artists, ["followers", "followers_count"])
        artist_pop_col = self.pick_column(artists, ["artist_popularity", "popularity"], required=False)

        rta_track_col = self.pick_column(rta, ["track_id", "trackId", "trackID"])
        rta_artist_col = self.pick_column(rta, ["artist_id", "artistId", "artistID"])

        self.column_mapping = {
            "tracks": {track_id_col: "track_id", track_pop_col: "track_popularity"},
            "artists": {artist_id_col: "artist_id", followers_col: "followers"},
            "r_track_artist": {rta_track_col: "track_id", rta_artist_col: "artist_id"},
        }
        if artist_pop_col:
            self.column_mapping["artists"][artist_pop_col] = "artist_popularity"

        tracks_norm = tracks[[track_id_col, track_pop_col]].rename(
            columns={track_id_col: "track_id", track_pop_col: "track_popularity"}
        )

        artists_cols = [artist_id_col, followers_col]
        rename_map = {artist_id_col: "artist_id", followers_col: "followers"}
        if artist_pop_col:
            artists_cols.append(artist_pop_col)
            rename_map[artist_pop_col] = "artist_popularity"
        artists_norm = artists[artists_cols].rename(columns=rename_map)

        rta_norm = rta[[rta_track_col, rta_artist_col]].rename(
            columns={rta_track_col: "track_id", rta_artist_col: "artist_id"}
        )

        return tracks_norm, artists_norm, rta_norm

    def prepare_analysis_data(self, tracks: pd.DataFrame, artists: pd.DataFrame, rta: pd.DataFrame) -> pd.DataFrame:
        avg_pop = (
            rta.merge(tracks, on="track_id", how="left")
            .groupby("artist_id", as_index=False)["track_popularity"]
            .mean()
            .rename(columns={"track_popularity": "avg_track_popularity"})
        )

        df = artists.merge(avg_pop, on="artist_id", how="left").dropna(subset=["avg_track_popularity"])

        df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0).clip(lower=0)
        df["followers_log1p"] = np.log1p(df["followers"])

        return df

    def create_scatter_plot(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(6, 4))
        sns.regplot(
            data=df,
            x="followers_log1p",
            y="avg_track_popularity",
            scatter_kws={"alpha": 0.4, "s": 18},
            line_kws={"linewidth": 2},
        )
        plt.title("Followers (log1p) vs Avg Track Popularity")
        plt.xlabel("followers_log1p")
        plt.ylabel("avg_track_popularity")

        self.save_plot("artist_followers_vs_avg_track_popularity.png", dpi=150)

    def calculate_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            "pearson_log": df[["followers_log1p", "avg_track_popularity"]].corr().iloc[0, 1],
            "spearman": df[["followers", "avg_track_popularity"]].corr(method="spearman").iloc[0, 1],
        }

    def execute(self, save_csv: bool = True) -> pd.DataFrame:
        self.logger.info("Running InfluenceAnalyzer")

        tracks, artists, rta = self.load_and_normalize_data()
        self.logger.info("Column mapping: %s", self.column_mapping)

        self.df = self.prepare_analysis_data(tracks, artists, rta)
        self.logger.info("Artists in analysis: %d", len(self.df))

        corr = self.calculate_correlation(self.df)
        self.logger.info("Correlation pearson_log=%.3f spearman=%.3f", corr["pearson_log"], corr["spearman"])

        self.create_scatter_plot(self.df)

        if save_csv:
            cols_out = ["artist_id", "followers", "avg_track_popularity", "followers_log1p"]
            if "artist_popularity" in self.df.columns:
                cols_out.insert(2, "artist_popularity")

            path = self.save_csv(self.df[cols_out], "artist_influence.csv", index=False, encoding="utf-8-sig")
            self.logger.info("Saved: %s", path)

        self.logger.info("Outputs in: %s", self.output_dir.resolve())
        return self.df