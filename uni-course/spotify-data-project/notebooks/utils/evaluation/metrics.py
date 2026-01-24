from __future__ import annotations

from typing import Dict

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

def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Zweck:
    - Berechnet Standard-Metriken für Regressionsmodelle.

    Metriken:
    - MAE  (Mean Absolute Error): durchschnittlicher absoluter Fehler
    - RMSE (Root Mean Squared Error): bestraft große Fehler stärker (Quadrat)
    - R2   (Bestimmtheitsmaß): erklärt wie viel Varianz durch das Modell erklärt wird

    Rückgabe:
    - Dictionary mit Metriken als float (praktisch für Logging/JSON).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_report_binary(y_true, y_proba, threshold=0.5) -> Dict[str, Any]:
    """
    Zweck:
    - Report für binäre Klassifikation, wenn das Modell Wahrscheinlichkeiten liefert.

    Parameter:
    - y_true: echte Labels (0/1)
    - y_proba: vorhergesagte Wahrscheinlichkeit für Klasse 1
    - threshold: Schwelle, ab wann 1 vorhergesagt wird (default 0.5)

    Output:
    - roc_auc: ROC-AUC (nur sinnvoll, wenn beide Klassen vorkommen)
    - pr_auc:  PR-AUC (Average Precision), oft wichtig bei Imbalance
    - f1:      F1-Score basierend auf threshold
    - confusion_matrix: [[TN, FP],[FN, TP]] als Liste (gut für JSON)

    Extra:
    - Wenn y_true nur eine Klasse enthält, sind AUC/F1 nicht sinnvoll -> None.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Aus Wahrscheinlichkeit wird harte Vorhersage 0/1 basierend auf Schwelle.
    y_pred = (y_proba >= threshold).astype(int)

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return out


def best_f1_threshold(y_true, proba, thresholds=np.linspace(0.05, 0.95, 19)):
    """
    Zweck:
    - Findet die Threshold-Schwelle, die den besten F1-Score liefert.

    Warum?
    - 0.5 ist nicht immer optimal (besonders bei unbalancierten Klassen).
    - Man scannt mehrere Thresholds und nimmt den besten nach F1.

    Parameter:
    - y_true: echte Labels (0/1)
    - proba: Modellwahrscheinlichkeiten für Klasse 1
    - thresholds: Liste/Array an Schwellenwerten (Default: 0.05..0.95)

    Rückgabe:
    - best_t: bester Threshold
    - best_f1: bester F1-Score, der damit erreicht wird
    """
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)

        # Wenn der neue F1 besser ist, merken wir uns die Schwelle.
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return float(best_t), float(best_f1)


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
