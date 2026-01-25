# utils/targets/cohort.py
from __future__ import annotations
import pandas as pd
import numpy as np

def build_cohort_targets(df: pd.DataFrame, *, cohort_col: str = "cohort_ym") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    pop = pd.to_numeric(df.get("popularity", np.nan), errors="coerce").astype("float64")

    # rank pct + mean by cohort (vectorized)
    cohort_ranks = df.groupby(cohort_col, sort=False)["popularity"].rank(pct=True) * 100
    cohort_means = df.groupby(cohort_col, sort=False)["popularity"].transform("mean")

    out["success_pct_in_cohort"] = cohort_ranks.astype("float64")
    out["success_residual_in_cohort"] = (pop - cohort_means).astype("float64")
    return out