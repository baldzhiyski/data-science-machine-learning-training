from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBClassifier
from ..splits import cohort_time_split
from ..metrics import binary_metrics, find_best_threshold_f1

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
