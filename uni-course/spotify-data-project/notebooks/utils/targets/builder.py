# utils/targets/builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from ..targets.config import CohortConfig, HitConfig, MoodConfig, ArtistTrajectoryConfig
from ..targets.preprocessing import ensure_release_cols, build_cohort_ym
from ..targets.cohort import build_cohort_targets
from ..targets.hit import build_hit_labels
from ..targets.mood import compute_mood_thresholds, build_mood_labels
from ..targets.artist_trajectory import build_artist_trajectory_targets

@dataclass(frozen=True)
class TargetsBuilderConfig:
    cohort: CohortConfig
    hit: HitConfig
    mood: MoodConfig
    traj: ArtistTrajectoryConfig

class TargetsBuilder:
    def __init__(self, cfg: TargetsBuilderConfig):
        self.cfg = cfg

    def build(self, track_df: pd.DataFrame, *, year_col: str = "release_year", month_col: str = "release_month") -> Dict[str, Any]:
        df = ensure_release_cols(track_df)

        # cohort_ym
        df["cohort_ym"] = build_cohort_ym(df, year_col, month_col)

        # (A) (B)
        cohort_out = build_cohort_targets(df, cohort_col="cohort_ym")
        df = df.join(cohort_out)

        y_success_pct = df["success_pct_in_cohort"].astype("float64")
        y_success_residual = df["success_residual_in_cohort"].astype("float64")

        # (C)
        y_hit = build_hit_labels(df, self.cfg.hit)

        # (D)
        mood_thresholds = compute_mood_thresholds(df, self.cfg.mood)
        Y_mood = build_mood_labels(df, self.cfg.mood, mood_thresholds)

        # (E)
        artist_panel, y_artist_growth, y_artist_breakout = build_artist_trajectory_targets(df, y_hit, self.cfg.traj)

        return {
            "track_df_with_targets": df,
            "y_success_pct": y_success_pct,
            "y_success_residual": y_success_residual,
            "y_hit": y_hit,
            "Y_mood": Y_mood,
            "mood_thresholds": mood_thresholds,
            "artist_panel": artist_panel,
            "y_artist_growth": y_artist_growth,
            "y_artist_breakout": y_artist_breakout,
        }