# utils/targets/hit.py
from __future__ import annotations
import numpy as np
import pandas as pd
from ..targets.config import HitConfig

def build_hit_labels(df: pd.DataFrame, cfg: HitConfig) -> pd.Series:
    pop = pd.to_numeric(df.get("popularity", np.nan), errors="coerce").astype("float64")
    year = pd.to_numeric(df.get("release_year", np.nan), errors="coerce").round().astype("Int64")

    nz = pop[(pop > 0) & pop.notna()]
    global_thr = float(nz.quantile(cfg.hit_percentile)) if len(nz) else 0.0

    year_pop = pd.DataFrame({"year": year, "pop": pop}).dropna(subset=["year", "pop"])
    if len(year_pop):
        year_counts = year_pop["year"].value_counts()
        good_years = year_counts[year_counts >= cfg.min_tracks_per_year].index
        good = year_pop[year_pop["year"].isin(good_years)]

        if len(good):
            year_thr = good.groupby("year")["pop"].quantile(cfg.hit_percentile)
            thr_map = year.map(year_thr).fillna(global_thr)
        else:
            thr_map = pd.Series(global_thr, index=df.index)
    else:
        thr_map = pd.Series(global_thr, index=df.index)

    y = (pop >= thr_map).fillna(False).astype("int8")

    # ensure minimum hit rate
    if float(y.mean()) < cfg.desired_rate:
        n = len(y)
        k = max(1, int(cfg.desired_rate * n))
        top_idx = pop.fillna(-np.inf).nlargest(k).index
        y = pd.Series(0, index=df.index, dtype="int8")
        y.loc[top_idx] = 1

    return y