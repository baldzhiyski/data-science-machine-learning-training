from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBRegressor
from ..splits import cohort_time_split
from ..metrics import regression_metrics
import optuna

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

    def tune(self, ds, n_trials: int = 50, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Success Residual (Regression).
        Optimiert MAE auf dem Validierungs-Split.
        """


        def _xgb_device_kwargs(dev: str):
            if dev.lower() in ("cuda", "gpu"):
                return {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "device": "cuda"}
            return {"tree_method": "hist", "predictor": "auto", "device": "cpu"}

        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
                "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 80.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 50.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            }

            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                **_xgb_device_kwargs(device),
                **params
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = model.predict(Xva)
            return regression_metrics(yva, pred)["MAE"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return {"best_params": study.best_params, "best_val_mae": float(study.best_value), "device": device}
