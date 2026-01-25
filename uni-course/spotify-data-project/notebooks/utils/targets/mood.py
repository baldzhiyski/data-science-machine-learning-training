# utils/targets/mood.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..targets.config import MoodConfig, MoodTagDef

def compute_mood_thresholds(df: pd.DataFrame, cfg: MoodConfig) -> Dict[str, float]:
    thr: Dict[str, float] = {}
    for tag in cfg.tags:
        if tag.col not in df.columns:
            thr[tag.name] = np.nan
            continue
        vals = pd.to_numeric(df[tag.col], errors="coerce").dropna()
        thr[tag.name] = float(vals.quantile(tag.q)) if len(vals) else np.nan
    return thr

def build_mood_labels(df: pd.DataFrame, cfg: MoodConfig, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, dtype="int8")

    for tag in cfg.tags:
        t = thresholds.get(tag.name, np.nan)
        if np.isnan(t) or tag.col not in df.columns:
            out[tag.name] = 0
            continue

        x = pd.to_numeric(df[tag.col], errors="coerce")
        if tag.direction == "gt":
            out[tag.name] = (x >= t).fillna(False).astype("int8")
        else:
            out[tag.name] = (x <= t).fillna(False).astype("int8")

    return out