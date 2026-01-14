from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .config import  MOOD_LABEL_SOURCE_AUDIO

"""
Dataset-Building (Preprocessing für Modell-Tasks).

Aufgabe:
- Stellt sicher, dass Features und Targets index-aligned sind
- Baut Task-spezifische Datasets (X, y, meta) mit konsistenter Filterung
- Enthält KEIN Training, KEINE Plots, nur Datenaufbereitung

Design-Entscheidung:
Wir arbeiten bewusst ohne `globals()`/Notebook-State.
Alle Funktionen erhalten Inputs explizit und geben Outputs explizit zurück.
"""


@dataclass
class TaskDataset:
    """
        Einheitliches Dataset-Format für Training und Evaluation.

        Attributes
        ----------
        X:
            Feature-Matrix (DataFrame), bereits gefiltert und index-reset, so dass Split-Indices stabil sind.
        y:
            Target (Series oder DataFrame für Multi-Label).
        meta:
            Metadaten pro Zeile (z.B. cohort_ym), nötig für zeit-/kohortenbasierte Splits.
        """
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame
    meta: pd.DataFrame  # must include cohort_ym for cohort splits


def _sanitize_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    # Keep simple + robust for parquet changes
    X = X.copy()
    # bool -> int
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    # keep numeric only
    X = X.select_dtypes(include=["number", "bool"]).copy()
    return X.fillna(0)


def build_success_pct_dataset(X_track: pd.DataFrame, track_df: pd.DataFrame, y_success_pct: pd.Series) -> TaskDataset:
    base_idx = X_track.index.intersection(track_df.index).intersection(y_success_pct.index)
    mask = y_success_pct.loc[base_idx].notna()

    X = _sanitize_numeric_df(X_track.loc[base_idx].loc[mask]).reset_index(drop=True)
    y = y_success_pct.loc[base_idx].loc[mask].astype(float).reset_index(drop=True)
    meta = track_df.loc[base_idx].loc[mask][["cohort_ym"]].reset_index(drop=True)

    return TaskDataset(X=X, y=y, meta=meta)


def build_success_residual_dataset(X_track: pd.DataFrame, track_df: pd.DataFrame, y_success_residual: pd.Series) -> TaskDataset:
    base_idx = X_track.index.intersection(track_df.index).intersection(y_success_residual.index)
    mask = y_success_residual.loc[base_idx].notna()

    X = _sanitize_numeric_df(X_track.loc[base_idx].loc[mask]).reset_index(drop=True)
    y = y_success_residual.loc[base_idx].loc[mask].astype("float32").reset_index(drop=True)
    meta = track_df.loc[base_idx].loc[mask][["cohort_ym"]].reset_index(drop=True)

    return TaskDataset(X=X, y=y, meta=meta)


def build_hit_dataset(X_track: pd.DataFrame, track_df: pd.DataFrame, y_hit: pd.Series) -> TaskDataset:
    base_idx = X_track.index.intersection(track_df.index).intersection(y_hit.index)
    y0 = y_hit.loc[base_idx].fillna(0).astype(int)

    X = _sanitize_numeric_df(X_track.loc[base_idx]).reset_index(drop=True)

    # extra safety: drop anything “popularity-like”
    leaky_cols = [c for c in X.columns if "popularity" in c.lower()]
    if leaky_cols:
        X = X.drop(columns=leaky_cols, errors="ignore")

    y = y0.reset_index(drop=True)
    meta = track_df.loc[base_idx][["cohort_ym"]].reset_index(drop=True)

    return TaskDataset(X=X, y=y, meta=meta)


def build_mood_dataset(X_track: pd.DataFrame, track_df: pd.DataFrame, Y_mood: pd.DataFrame) -> TaskDataset:
    base_idx = X_track.index.intersection(track_df.index).intersection(Y_mood.index)
    mask = Y_mood.loc[base_idx].notna().all(axis=1)

    X = _sanitize_numeric_df(X_track.loc[base_idx].loc[mask]).reset_index(drop=True)
    # leakage guard (same as your notebook)
    X = X.drop(columns=[c for c in MOOD_LABEL_SOURCE_AUDIO if c in X.columns], errors="ignore")

    Y = Y_mood.loc[base_idx].loc[mask].astype(int).reset_index(drop=True)
    meta = track_df.loc[base_idx].loc[mask][["cohort_ym"]].reset_index(drop=True)

    return TaskDataset(X=X, y=Y, meta=meta)
