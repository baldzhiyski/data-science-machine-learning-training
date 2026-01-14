from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, f1_score, confusion_matrix
)

"""
Metriken & Schwellenwert-Logik.

Enthält:
- Regression: MAE/RMSE/R2
- Binary: ROC-AUC, PR-AUC, F1, Confusion Matrix
- Threshold-Suche: bestes F1 auf Val

Hinweis:
Metrics sind zentralisiert, damit alle Tasks konsistent berichten.
"""



def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def find_best_threshold_f1(y_true, y_proba, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, float(best_f1)


def binary_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return out
