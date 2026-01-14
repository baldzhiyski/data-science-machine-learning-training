from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBRegressor
from ..splits import cohort_time_split
from ..metrics import regression_metrics

"""
Task: Success Residual within Cohort (Regression).

Ziel:
Modelliert "Überperformance" relativ zur Kohorte (Residual statt roher Erfolg).

Hinweis:
Residual-Targets sind oft noisier -> stärkere Regularisierung sinnvoll.
Negative R² auf Val/Test ist ein typisches Symptom für Overfitting oder Target-/Split-Bugs.
"""


@dataclass
class SuccessResidualTrainer:
    seed: int = 42

    def fit_eval(self, ds):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X[idx_tr], ds.y[idx_tr]
        Xva, yva = ds.X[idx_va], ds.y[idx_va]
        Xte, yte = ds.X[idx_te], ds.y[idx_te]

        # More regularized than success_pct (residual target is noisier)
        model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5.0,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        metrics = {
            "train": regression_metrics(ytr, model.predict(Xtr)),
            "val": regression_metrics(yva, model.predict(Xva)),
            "test": regression_metrics(yte, model.predict(Xte)),
        }
        return model, metrics
