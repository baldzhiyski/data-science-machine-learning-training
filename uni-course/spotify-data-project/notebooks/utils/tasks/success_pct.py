from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBRegressor
from ..splits import cohort_time_split
from ..metrics import regression_metrics
import optuna

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

    def fit_eval(self, ds,params : dict | None =None):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X[idx_tr], ds.y[idx_tr]
        Xva, yva = ds.X[idx_va], ds.y[idx_va]
        Xte, yte = ds.X[idx_te], ds.y[idx_te]

        if params:
            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                **params
            )
        else:
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

    def tune(self, ds, n_trials: int = 40, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Success Percentile (Regression).
        Optimiert MAE auf dem Validierungs-Split.

        Improvements:
        - Do NOT tune n_estimators; use high cap + early stopping
        - Stronger regularization + lower capacity to reduce time-split overfitting
        - Objective aligned with MAE
        """

        from xgboost import XGBRegressor
        import optuna
        import numpy as np

        def _xgb_device_kwargs(dev: str | None):
            dev = (dev or "cpu").lower().strip()
            if dev == "gpu":
                dev = "cuda"
            if dev.startswith("cuda"):
                return {"tree_method": "hist", "device": dev}
            return {"tree_method": "hist", "device": "cpu"}

        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        def objective(trial):
            params = {
                # Let early stopping choose the best number of trees
                "n_estimators": 20000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

                # Capacity control
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 40.0, log=True),

                # Subsampling improves generalization
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),

                # Meaningful regularization (avoid near-zero)
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),

                # Small gamma range (too large can underfit)
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),

                # Hist regularizer (optional but powerful)
                "max_leaves": trial.suggest_int("max_leaves", 16, 256),
            }

            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                objective="reg:absoluteerror",  # aligns with MAE
                eval_metric="mae",
                early_stopping_rounds=300,
                **_xgb_device_kwargs(device),
                **params,
            )

            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            # Use best_iteration if available
            if getattr(model, "best_iteration", None) is not None:
                pred = model.predict(Xva, iteration_range=(0, model.best_iteration + 1))
            else:
                pred = model.predict(Xva)

            mae = regression_metrics(yva, pred)["MAE"]
            return float(mae)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": dict(study.best_params),
            "best_val_mae": float(study.best_value),
            "device": device,
        }
