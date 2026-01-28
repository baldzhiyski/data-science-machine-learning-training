from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


"""
Datenladen (I/O Layer).

Aufgabe:
- Lädt alle vorberechneten Parquet/CSV Dateien (Features + Targets + Meta-DFs)
- Gibt ein DataBundle zurück, damit nachgelagerte Schritte nicht über globale Variablen arbeiten.

Wichtig:
- Keine Modelllogik hier
- Keine Feature-Masken/Splits hier
"""


@dataclass
class DataBundle:
    track_df: pd.DataFrame
    artist_df: pd.DataFrame
    album_df: pd.DataFrame



    X_track_hit:  pd.DataFrame
    X_track_mood: pd.DataFrame
    X_track_success: pd.DataFrame
    X_album: pd.DataFrame
    X_artist: pd.DataFrame
    y_artist_growth: pd.Series
    y_artist_breakout: pd.Series

    y_success_pct: pd.Series
    y_success_residual: pd.Series
    y_hit: pd.Series
    Y_mood: pd.DataFrame

    artist_panel: pd.DataFrame | None

    track_genre_df: pd.DataFrame | None
    artist_genre_df: pd.DataFrame | None
    album_genre_df: pd.DataFrame | None


def _read_series(path: Path) -> pd.Series:
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 1:
        return obj.squeeze("columns")
    if isinstance(obj, pd.Series):
        return obj
    raise ValueError(f"Expected 1-col parquet series at {path}, got {type(obj)} with shape {getattr(obj, 'shape', None)}")


def load_all(input_targets_path: Path) -> DataBundle:
    """
        Lädt alle benötigten Inputs für Training/Reports.

        Parameter
        ---------
        input_targets_path:
            Ordner, in dem die vorbereiteten Parquet-Dateien liegen.

        Returns
        -------
        DataBundle:
            Container mit track_df/artist_df/album_df, X_track und allen Targets (y_*).
        """
    track_df = pd.read_parquet(input_targets_path / "track_df.parquet")
    artist_df = pd.read_parquet(input_targets_path / "artist_df.parquet")
    album_df = pd.read_parquet(input_targets_path / "album_df.parquet")

    X_track_hit = pd.read_parquet(input_targets_path / "X_track_hit.parquet")
    X_track_mood = pd.read_parquet(input_targets_path / "X_track_mood.parquet")
    X_track_success = pd.read_parquet(input_targets_path / "X_track_success.parquet")
    X_track_mood = pd.read_parquet(input_targets_path / "X_track_mood.parquet")
    X_album = pd.read_parquet(input_targets_path / "X_album.parquet")
    X_artist = pd.read_parquet(input_targets_path / "X_artist.parquet")

    y_success_pct = _read_series(input_targets_path / "y_success_pct.parquet")
    y_success_residual = _read_series(input_targets_path / "y_success_residual.parquet")
    y_hit = _read_series(input_targets_path / "y_hit.parquet")
    y_artist_growth = _read_series(input_targets_path / "y_artist_growth.parquet")
    y_artist_breakout = _read_series(input_targets_path / "y_artist_breakout.parquet")

    Y_mood = pd.read_parquet(input_targets_path / "Y_mood.parquet")

    artist_panel_path = input_targets_path / "artist_panel.parquet"
    artist_panel = pd.read_parquet(artist_panel_path) if artist_panel_path.exists() else None

    def maybe_read(name: str) -> pd.DataFrame | None:
        p = input_targets_path / name
        return pd.read_parquet(p) if p.exists() else None

    track_genre_df = maybe_read("track_genre_multihot.parquet")
    artist_genre_df = maybe_read("artist_genre_multihot.parquet")
    album_genre_df = maybe_read("album_genre_multihot.parquet")

    return DataBundle(
        track_df=track_df,
        artist_df=artist_df,
        album_df=album_df,
        X_track_hit =X_track_hit,
        X_track_mood=X_track_mood,
        X_track_success=X_track_success,
        X_album=X_album,
        X_artist=X_artist,
        y_artist_growth=y_artist_growth,
        y_artist_breakout=y_artist_breakout,
        y_success_pct=y_success_pct,
        y_success_residual=y_success_residual,
        y_hit=y_hit,
        Y_mood=Y_mood,
        artist_panel=artist_panel,
        track_genre_df=track_genre_df,
        artist_genre_df=artist_genre_df,
        album_genre_df=album_genre_df,
    )
