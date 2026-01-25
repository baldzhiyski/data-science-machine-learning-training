# utils/targets/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class HitConfig:
    hit_percentile: float = 0.90
    desired_rate: float = 0.10
    min_tracks_per_year: int = 200

@dataclass(frozen=True)
class MoodTagDef:
    name: str
    col: str
    q: float
    direction: str  # "gt" or "lt"

@dataclass(frozen=True)
class MoodConfig:
    tags: List[MoodTagDef]

@dataclass(frozen=True)
class CohortConfig:
    modern_year_threshold: int = 2015  

@dataclass(frozen=True)
class ArtistTrajectoryConfig:
    past_m: int = 6
    future_m: int = 6
    min_past_tracks: int = 5
    breakout_q: float = 0.90
    audio_cols_limit: int = 2