from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from ..splits import cohort_time_split
import optuna

"""
Task: Mood/Genre Multi-Label Classification.
Ziel:
Vorhersage mehrerer Stimmungs- oder Genre-Labels pro Track.
Hinweise:
- Multi-Label Klassifikation mit unabhängigen Labels.
- Per-Label Threshold-Optimierung zur Verbesserung der F1-Metriken.
"""

@dataclass
class MoodTrainer:
    seed: int = 42

    def fit_eval(self, ds):
        # keep it simple: random split (you can swap to cohort split later if you want)
        Xtr, Xte, Ytr, Yte = train_test_split(
            ds.X, ds.y, test_size=0.20, random_state=self.seed
        )

        base = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            class_weight="balanced",
            random_state=self.seed,
        )
        clf = OneVsRestClassifier(base)
        clf.fit(Xtr, Ytr)

        # probabilities
        P = clf.predict_proba(Xte)

        # per-label threshold search (simple)
        thresholds = {}
        Y_true = Yte.values
        for j, label in enumerate(ds.y.columns):
            best_thr, best_f1 = 0.5, -1
            for thr in np.linspace(0.05, 0.95, 19):
                pred = (P[:, j] >= thr).astype(int)
                f1 = f1_score(Y_true[:, j], pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            thresholds[label] = best_thr

        Y_pred = np.zeros_like(Y_true)
        for j, label in enumerate(ds.y.columns):
            Y_pred[:, j] = (P[:, j] >= thresholds[label]).astype(int)

        micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
        macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
        per_label = {
            label: float(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0))
            for j, label in enumerate(ds.y.columns)
        }

        metrics = {
            "micro_f1": float(micro),
            "macro_f1": float(macro),
            "per_label_f1": per_label,
        }
        return clf, metrics, thresholds

    def tune(self, ds, n_trials: int = 30):
        """
        Hyperparameter-Tuning für Mood Multi-Label.
        Optimiert micro-F1 auf dem Validierungs-Split.

        Hinweis:
        Hier typischerweise CPU (sklearn). GPU bringt i.d.R. nichts.
        """


        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, Ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, Yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
            loss = trial.suggest_categorical("loss", ["log_loss", "modified_huber"])
            penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
            l1_ratio = 0.15
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)

            base = SGDClassifier(
                random_state=self.seed,
                alpha=alpha,
                loss=loss,
                penalty=penalty,
                l1_ratio=l1_ratio if penalty == "elasticnet" else None,
                max_iter=3000,
                n_jobs=4,
            )
            model = OneVsRestClassifier(base, n_jobs=4)
            model.fit(Xtr, Ytr)


            proba = np.column_stack([est.predict_proba(Xva)[:, 1] for est in model.estimators_])
            pred = (proba >= 0.5).astype(int)

            return f1_score(Yva.values, pred, average="micro")  # maximize

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return {"best_params": study.best_params, "best_val_micro_f1": float(study.best_value)}
