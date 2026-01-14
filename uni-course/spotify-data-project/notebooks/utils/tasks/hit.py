from __future__ import annotations
from dataclasses import dataclass

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

    def fit_eval(self, ds):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X[idx_tr], ds.y[idx_tr]
        Xva, yva = ds.X[idx_va], ds.y[idx_va]
        Xte, yte = ds.X[idx_te], ds.y[idx_te]

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
        Optimiert PR-AUC auf dem Validierungs-Split.
        """


        def _xgb_device_kwargs(dev: str):
            if dev.lower() in ("cuda", "gpu"):
                return {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "device": "cuda"}
            return {"tree_method": "hist", "predictor": "auto", "device": "cpu"}

        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        # Klassenbalance optional (hilft manchmal)
        pos = float(np.sum(ytr == 1))
        neg = float(np.sum(ytr == 0))
        base_spw = (neg / pos) if pos > 0 else 1.0

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 80.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5 * base_spw, 2.0 * base_spw),
            }

            clf = XGBClassifier(
                random_state=self.seed,
                n_jobs=4,
                eval_metric="aucpr",
                **_xgb_device_kwargs(device),
                **params
            )
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            proba = clf.predict_proba(Xva)[:, 1]
            return average_precision_score(yva, proba)  # maximize

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return {"best_params": study.best_params, "best_val_pr_auc": float(study.best_value), "device": device}
