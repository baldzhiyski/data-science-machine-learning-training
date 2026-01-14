from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from ..splits import time_fraction_split
from ..metrics import regression_metrics

"""
Task: Artist Trajectory (Growth & Breakout).

Inputs:
artist_panel mit monatlichen Aggregaten pro Artist.

Targets:
- y_growth: Wachstum (Regression; häufig log1p-transformiert)
- y_breakout: Breakout-Event (Classification)

Wichtig:
- Sortierung nach Zeitspalte ist Pflicht, damit der Split zeitlich konsistent ist.
- Drop von IDs/Targets/Time aus Features, um Leakage zu vermeiden.
"""


@dataclass
class ArtistTrajectoryTrainer:
    seed: int = 42

    # map your actual column names here
    time_col: str = "release_month_ts"
    growth_col: str = "y_growth"
    breakout_col: str = "y_breakout"
    # columns to never use as features
    drop_cols: tuple = ("artist_id",)

    def fit_eval(self, artist_panel: pd.DataFrame):
        # --- validate required columns exist
        required = {self.time_col, self.growth_col, self.breakout_col}
        missing = required - set(artist_panel.columns)
        if missing:
            raise ValueError(
                f"artist_panel missing columns: {missing}. "
                f"Available: {list(artist_panel.columns)}"
            )

        # --- sort by time (important for time split)
        df = artist_panel.sort_values(self.time_col).reset_index(drop=True)

        # --- targets
        y_growth_raw = df[self.growth_col].astype(float).values
        y_growth = np.log1p(np.clip(y_growth_raw, a_min=0, a_max=None))  # safer
        y_break = df[self.breakout_col].astype(int).values

        # --- features
        X = df.select_dtypes(include=["number", "bool"]).copy()

        # drop targets + identifiers + time col from features
        X = X.drop(columns=[self.growth_col, self.breakout_col, self.time_col], errors="ignore")
        X = X.drop(columns=list(self.drop_cols), errors="ignore")
        X = X.fillna(0)

        # --- split by time order
        idx_tr, idx_va, idx_te = time_fraction_split(
            len(df), val_frac=0.10, test_frac=0.15, min_val=30, min_test=30
        )

        Xtr, Xva, Xte = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]
        ytr_g, yva_g, yte_g = y_growth[idx_tr], y_growth[idx_va], y_growth[idx_te]
        ytr_b, yva_b, yte_b = y_break[idx_tr], y_break[idx_va], y_break[idx_te]

        # --- growth regressor
        reg = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        reg.fit(Xtr, ytr_g, eval_set=[(Xva, yva_g)], verbose=False)

        growth_metrics = regression_metrics(yte_g, reg.predict(Xte))
        growth_metrics = {
            "MAE_log": growth_metrics["MAE"],
            "RMSE_log": growth_metrics["RMSE"],
            "R2_log": growth_metrics["R2"],
            "n_test": int(len(yte_g)),
        }

        # --- breakout classifier
        clf = XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            eval_metric="aucpr",
        )
        clf.fit(Xtr, ytr_b, eval_set=[(Xva, yva_b)], verbose=False)

        proba = clf.predict_proba(Xte)[:, 1]
        breakout_metrics = {
            "ROC_AUC": float(roc_auc_score(yte_b, proba)) if len(np.unique(yte_b)) > 1 else float("nan"),
            "PR_AUC": float(average_precision_score(yte_b, proba)) if len(np.unique(yte_b)) > 1 else float("nan"),
            "breakout_rate_test": float(yte_b.mean()),
            "n_test": int(len(yte_b)),
        }

        return {"growth_model": reg, "breakout_model": clf}, growth_metrics, breakout_metrics
