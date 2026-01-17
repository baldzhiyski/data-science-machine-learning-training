from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from ..splits import cohort_time_split
from ..metrics import binary_metrics, find_best_threshold_f1
import optuna
"""
Task: Hit Prediction (Binary Classification).

Ziel:
Vorhersage, ob ein Track als "Hit" gilt (y_hit).

Besonderheiten:
- Klassifikationsproblem mit meist unausgeglichenen Klassen
- Evaluation sollte PR-AUC berücksichtigen (besser als ROC-AUC bei Imbalance)
- Best-Threshold wird auf Validierung optimiert (F1)
"""


@dataclass
class HitTrainer:
    """
        Trainer für Hit Prediction.

        Methoden
        --------
        fit_eval(ds):
            Trainiert auf Train-Split, optimiert Threshold auf Val, reportet Metriken auf Test.
        """
    seed: int = 42

    def fit_eval(self, ds,params : dict | None = None):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X[idx_tr], ds.y[idx_tr]
        Xva, yva = ds.X[idx_va], ds.y[idx_va]
        Xte, yte = ds.X[idx_te], ds.y[idx_te]

        if params:
            clf = XGBClassifier(
                random_state=self.seed,
                n_jobs=4,
                **params
            )
        else:

            clf = XGBClassifier(
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=self.seed,
                n_jobs=4,
                eval_metric="aucpr",
            )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        proba_va = clf.predict_proba(Xva)[:, 1]
        thr, thr_f1 = find_best_threshold_f1(yva, proba_va)

        proba_te = clf.predict_proba(Xte)[:, 1]
        m = binary_metrics(yte, proba_te, threshold=thr)
        m["best_threshold"] = float(thr)
        m["best_threshold_f1_on_val"] = float(thr_f1)
        return clf, m, thr

    def tune(self, ds, n_trials: int = 60, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Hit Prediction (Binary).
        Optimiert PR-AUC (Average Precision) auf dem Validierungs-Split.
        Uses early stopping + high n_estimators cap (do NOT tune n_estimators directly).
        """

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

        # base class imbalance
        pos = float(np.sum(ytr == 1))
        neg = float(np.sum(ytr == 0))
        base_spw = (neg / pos) if pos > 0 else 1.0

        def objective(trial):
            params = {
                # Let early stopping choose the best iteration.
                "n_estimators": 20000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

                # Reduce capacity (depth 10 is too much for temporal generalization)
                "max_depth": trial.suggest_int("max_depth", 3, 7),

                # Sampling helps generalization; avoid always 1.0
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),

                # Prevent tiny leaves (1e-3 is overfit-friendly)
                "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 30.0, log=True),

                # Make regularization meaningful (don’t start at 1e-8)
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),

                # Gamma too large often just kills splits; keep smaller
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),

                # Imbalance
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 0.8 * base_spw, 1.6 * base_spw
                ),

                # Optional stabilizer for imbalanced logistic
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            }

            clf = XGBClassifier(
                random_state=self.seed,
                n_jobs=4,
                early_stopping_rounds=300,  # more patient; your n is large
                eval_metric="aucpr",
                **_xgb_device_kwargs(device),
                **params,
            )

            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            proba = clf.predict_proba(Xva)[:, 1]
            return average_precision_score(yva, proba)  # maximize PR-AUC

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_val_pr_auc": float(study.best_value),
            "device": device,
        }
