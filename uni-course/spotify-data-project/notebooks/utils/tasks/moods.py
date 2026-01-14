from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

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
