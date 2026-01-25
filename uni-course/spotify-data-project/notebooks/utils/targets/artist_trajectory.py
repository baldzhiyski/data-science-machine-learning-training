
from __future__ import annotations
from ..targets.config import ArtistTrajectoryConfig
from ..targets.preprocessing import ensure_release_month_ts

import json
import numpy as np
import pandas as pd

def _ensure_list_fast(x):
    # missing
    if x is None:
        return []
    # pd.NA / NaN
    if isinstance(x, float) and np.isnan(x):
        return []
    if x is pd.NA:
        return []

    # already list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return list(x)

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # pandas Series (rare but possible)
    if isinstance(x, pd.Series):
        return x.dropna().tolist()

    # string: maybe JSON like '["id1","id2"]' or 'id1,id2'
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s[0] == "[" and s[-1] == "]":
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                pass
        # fallback: comma-separated
        return [p.strip() for p in s.split(",") if p.strip()]

    # fallback: single value
    return [x]
def build_artist_trajectory_targets(
    track_df: pd.DataFrame,
    y_hit: pd.Series,
    cfg: ArtistTrajectoryConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = track_df.copy()

    if "release_month_ts" not in df.columns:
        df["release_month_ts"] = ensure_release_month_ts(df)

    # audio cols limited
    audio_cols = [c for c in ["danceability", "energy", "valence", "tempo"] if c in df.columns]
    audio_cols = audio_cols[: cfg.audio_cols_limit]

    track_id_col = "track_id" if "track_id" in df.columns else "id"
    if track_id_col not in df.columns:
        raise ValueError("Need track_id or id")

    df["_hit"] = y_hit.astype("int8")

    base_cols = [track_id_col, "release_month_ts", "popularity", "_hit"] + audio_cols
    base = df[base_cols].copy()
    base.columns = ["track_id", "release_month_ts", "popularity", "hit"] + audio_cols
    base["track_id"] = base["track_id"].astype(str)
    base["popularity"] = pd.to_numeric(base["popularity"], errors="coerce")
    base["hit"] = base["hit"].astype("int8")
    for c in audio_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    base = base.dropna(subset=["track_id", "release_month_ts", "popularity"]).drop_duplicates("track_id")

    # mapping track -> artists
    if "artist_id" in df.columns:
        ta = df[[track_id_col, "artist_id"]].copy()
        ta.columns = ["track_id", "artist_id"]
    elif "artist_ids" in df.columns:
        ta = df[[track_id_col, "artist_ids"]].copy()
        ta.columns = ["track_id", "artist_ids"]
        ta["track_id"] = ta["track_id"].astype(str)
        ta["artist_ids"] = ta["artist_ids"].apply(_ensure_list_fast)
        ta = ta.explode("artist_ids").rename(columns={"artist_ids": "artist_id"})
    else:
        raise ValueError("Need 'artist_id' or 'artist_ids'")

    ta["track_id"] = ta["track_id"].astype(str)
    ta["artist_id"] = ta["artist_id"].astype(str)

    # weights per track (split credit between artists)
    n_artists = ta.groupby("track_id")["artist_id"].size()
    ta["w"] = ta["track_id"].map(1.0 / n_artists)

    traj = ta.merge(base, on="track_id", how="inner")
    traj["w_pop"] = traj["w"] * traj["popularity"]
    traj["w_hit"] = traj["w"] * traj["hit"]
    traj["artist_id"] = traj["artist_id"].astype("category")

    # monthly agg
    artist_month = traj.groupby(["artist_id", "release_month_ts"], sort=False, observed=True).agg({
        "track_id": "nunique",
        "w": "sum",
        "w_pop": "sum",
        "popularity": "max",
        "w_hit": "sum"
    }).reset_index()

    artist_month.columns = [
        "artist_id", "release_month_ts", "n_unique_tracks_month", "n_tracks_month",
        "pop_sum_month", "pop_max_month", "hit_sum_month"
    ]
    artist_month["pop_mean_month"] = artist_month["pop_sum_month"] / artist_month["n_tracks_month"]
    artist_month = artist_month.sort_values(["artist_id", "release_month_ts"]).reset_index(drop=True)

    g = artist_month.groupby("artist_id", sort=False, observed=True)

    # past rolling
    P, F = cfg.past_m, cfg.future_m
    artist_month["past_unique_tracks"] = g["n_unique_tracks_month"].transform(lambda s: s.rolling(P, min_periods=1).sum())
    artist_month["past_pop_mean"] = g["pop_mean_month"].transform(lambda s: s.rolling(P, min_periods=1).mean())
    artist_month["past_hit_sum"] = g["hit_sum_month"].transform(lambda s: s.rolling(P, min_periods=1).sum())

    # future (shift + rolling)
    artist_month["future_pop_mean"] = g["pop_mean_month"].transform(lambda s: s.shift(-F).rolling(F, min_periods=1).mean())
    artist_month["future_tracks"] = g["n_tracks_month"].transform(lambda s: s.shift(-F).rolling(F, min_periods=1).sum())

    panel = artist_month[
        (artist_month["past_unique_tracks"] >= cfg.min_past_tracks) &
        artist_month["future_tracks"].notna() & (artist_month["future_tracks"] > 0)
    ].copy().reset_index(drop=True)

    panel["y_growth"] = panel["future_pop_mean"] - panel["past_pop_mean"]
    panel["year"] = panel["release_month_ts"].dt.year

    panel["y_breakout"] = panel.groupby("year")["y_growth"].transform(
        lambda x: (x >= x.quantile(cfg.breakout_q)).astype("int8")
    )

    y_growth = panel["y_growth"].astype("float64")
    y_breakout = panel["y_breakout"].astype("int8")
    return panel, y_growth, y_breakout