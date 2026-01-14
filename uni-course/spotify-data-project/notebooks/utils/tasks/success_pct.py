from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBRegressor
from ..splits import cohort_time_split
from ..metrics import regression_metrics


"""
Task: Success Percentile Prediction (Regression).
Ziel:
Vorhersage des Erfolgpercentils (0-100) eines Tracks innerhalb seiner Kohorte.
Hinweise:
- Kohortenbasierte Zeit-Splits sind wichtig, um Daten-Leakage zu vermeiden.
- Erfolgpercentile sind oft schief verteilt; geeignete Metriken wählen.
"""

@dataclass
class SuccessPctTrainer:
    seed: int = 42

    def fit_eval(self, ds):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X[idx_tr], ds.y[idx_tr]
        Xva, yva = ds.X[idx_va], ds.y[idx_va]
        Xte, yte = ds.X[idx_te], ds.y[idx_te]

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        pred_te = model.predict(Xte)
        m = regression_metrics(yte, pred_te)
        m.update({
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "n_test": int(len(yte)),
            "label_range_expected": [0, 100],
        })
        return model, m
