"""
Einheitliche Tuning-Utilities für Hyperparameter-Optimierung.

Dieses Modul bietet konsistente:
- Optuna-Study-Erstellung mit reproduzierbarem Seeding
- XGBoost-Gerätekonfiguration (CPU/GPU)
- Early-Stopping-Strategie
- Suchraum-Builder mit realistischen Constraints
- Artefakt-Sammlung und Reporting

Verwendung:
    from .tuning_utils import (
        create_optuna_study,
        xgb_device_kwargs,
        XGB_BASE_PARAMS,
        collect_tuning_artifacts,
    )

Design-Prinzipien:
-----------------
1. Reproduzierbarkeit: Feste Sampler-Seeds gewährleisten reproduzierbare Trials.
2. Anti-Overfit: Suchräume beschränken Kapazität (Tiefe, Blätter) und
   erzwingen sinnvolle Regularisierung für bessere Zeit-Split-Generalisierung.
3. Early Stopping: Immer hohe n_estimators-Obergrenze mit Early Stopping;
   n_estimators niemals direkt tunen.
4. Metrik-Alignment: XGBoost eval_metric entspricht dem Optimierungsziel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler

# Optuna INFO-Logs unterdrücken für saubere Ausgabe
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# Konstanten: Einheitliche Suchraum-Constraints
# =============================================================================

# Basis-XGBoost-Parameter (verwendet wenn kein Tuning)
XGB_BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5.0,
    "gamma": 0.0,
}

# Einheitliche Early-Stopping-Konfiguration
EARLY_STOPPING_ROUNDS = 300
N_ESTIMATORS_CAP = 20000


# =============================================================================
# Optuna-Study-Factory
# =============================================================================

def create_optuna_study(
    direction: str = "minimize",
    seed: int = 42,
    study_name: Optional[str] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
) -> optuna.Study:
    """
    Erstellt eine Optuna-Study mit reproduzierbarem Sampling.

    Parameter
    ---------
    direction : str
        'minimize' oder 'maximize'
    seed : int
        Random-Seed für TPE-Sampler (Reproduzierbarkeit)
    study_name : str, optional
        Name der Study (nützlich für Logging)
    pruner : optuna.pruners.BasePruner, optional
        Pruner für vorzeitiges Trial-Abbrechen

    Rückgabe
    --------
    optuna.Study
        Konfigurierte Study bereit zur Optimierung
    """
    sampler = TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)

    if pruner is None:
        # Median-Pruner mit Geduld für baumbasierte Modelle
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=100,
            interval_steps=50,
        )

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )
    return study


# =============================================================================
# XGBoost-Gerätekonfiguration
# =============================================================================

def xgb_device_kwargs(device: str = "cpu") -> Dict[str, str]:
    """
    Gibt XGBoost-kwargs für Gerät (CPU/GPU) zurück.

    Parameter
    ---------
    device : str
        'cpu', 'gpu', 'cuda', oder 'cuda:0', etc.

    Rückgabe
    --------
    dict
        {"tree_method": "hist", "device": <device>}
    """
    dev = (device or "cpu").lower().strip()
    if dev == "gpu":
        dev = "cuda"
    if dev.startswith("cuda"):
        return {"tree_method": "hist", "device": dev}
    return {"tree_method": "hist", "device": "cpu"}


# =============================================================================
# Suchraum-Builder (Realistische Constraints)
# =============================================================================

def suggest_xgb_regression_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Schlägt XGBoost-Hyperparameter für Regressions-Tasks vor.

    Strategie:
    - Hohe n_estimators-Obergrenze mit Early Stopping (nicht getuned)
    - Tiefe/Blätter beschränken um Overfitting bei Zeit-Splits zu verhindern
    - Sinnvolle Regularisierungsbereiche (nicht nahe Null)
    - Moderates Subsampling für Generalisierung
    """
    return {
        "n_estimators": N_ESTIMATORS_CAP,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

        # Kapazitätskontrolle (niedrigere max_depth für Zeit-Split-Generalisierung)
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 50.0, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 16, 128),

        # Subsampling (1.0 vermeiden um Varianz zu reduzieren)
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),

        # Regularisierung (sinnvolle Bereiche)
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),

        # Split-Penalty (moderater Bereich)
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }


def suggest_xgb_classification_params(
    trial: optuna.Trial,
    base_scale_pos_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Schlägt XGBoost-Hyperparameter für binäre Klassifikation vor.

    Strategie:
    - Gleiche Kapazitäts-/Regularisierungs-Constraints wie Regression
    - scale_pos_weight um das Klassen-Ungleichgewicht herum getuned
    - max_delta_step für numerische Stabilität bei unbalanciertem Logistic-Loss
    """
    params = suggest_xgb_regression_params(trial)

    # Ungleichgewichts-Behandlung: um Basis-Ratio herum tunen (keine extremen Bereiche)
    params["scale_pos_weight"] = trial.suggest_float(
        "scale_pos_weight",
        0.7 * base_scale_pos_weight,
        1.5 * base_scale_pos_weight,
    )

    # Stabilisator für unbalanciertes Logistic
    params["max_delta_step"] = trial.suggest_int("max_delta_step", 0, 5)

    return params


def suggest_xgb_ranker_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Schlägt XGBoost-Hyperparameter für Ranking-Tasks vor.

    Strategie:
    - Ähnlich wie Regression aber leicht höhere Kapazität erlaubt
    - Ranking ist toleranter gegenüber Tiefe (Pairwise/Listwise-Losses)
    """
    return {
        "n_estimators": N_ESTIMATORS_CAP,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),

        # Leicht höhere max_depth für Ranking erlaubt
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 60.0, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 16, 256),

        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),

        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }


def suggest_sgd_multilabel_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Schlägt SGDClassifier-Hyperparameter für Multi-Label-Klassifikation vor.

    Strategie:
    - Alpha (Regularisierung) über weiten Bereich tunen
    - Loss-Funktion-Auswahl (log_loss vs modified_huber)
    - Penalty-Typ mit passendem l1_ratio für Elasticnet
    - Globales Threshold-Tuning (großer Gewinn für micro-F1)
    """
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    params = {
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "loss": trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
        "penalty": penalty,
        "l1_ratio": None,
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)

    return params


# =============================================================================
# Tuning-Artefakt-Sammlung
# =============================================================================

@dataclass
class TuningResult:
    """
    Container für Tuning-Ergebnisse mit konsistenter Struktur.

    Attribute
    ---------
    best_params : dict
        Beste gefundene Hyperparameter
    best_score : float
        Bester Validierungs-Score (primäre Metrik)
    best_iteration : int, optional
        Beste Boosting-Iteration (wenn Early Stopping verwendet)
    metric_name : str
        Name der optimierten Metrik
    direction : str
        'minimize' oder 'maximize'
    n_trials : int
        Anzahl abgeschlossener Trials
    study_name : str, optional
        Name der Optuna-Study
    extra_metrics : dict
        Zusätzliche während Tuning aufgezeichnete Metriken
    device : str
        Für Training verwendetes Gerät
    """
    best_params: Dict[str, Any]
    best_score: float
    metric_name: str
    direction: str
    n_trials: int
    best_iteration: Optional[int] = None
    study_name: Optional[str] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Serialisierung."""
        return {
            "best_params": self.best_params,
            f"best_val_{self.metric_name}": self.best_score,
            "best_iteration": self.best_iteration,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "extra_metrics": self.extra_metrics,
            "device": self.device,
        }


def collect_tuning_artifacts(
    study: optuna.Study,
    metric_name: str,
    device: str = "cpu",
    best_iteration: Optional[int] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> TuningResult:
    """
    Sammelt Tuning-Artefakte aus abgeschlossener Optuna-Study.

    Parameter
    ---------
    study : optuna.Study
        Abgeschlossene Optuna-Study
    metric_name : str
        Name der optimierten Metrik (z.B. 'mae', 'pr_auc')
    device : str
        Für Training verwendetes Gerät
    best_iteration : int, optional
        Beste Boosting-Iteration von Early Stopping
    extra_metrics : dict, optional
        Zusätzliche einzuschließende Metriken

    Rückgabe
    --------
    TuningResult
        Strukturiertes Tuning-Ergebnis
    """
    return TuningResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        metric_name=metric_name,
        direction=study.direction.name.lower(),
        n_trials=len(study.trials),
        best_iteration=best_iteration,
        study_name=study.study_name,
        extra_metrics=extra_metrics or {},
        device=device,
    )


# =============================================================================
# Multi-Label-Threshold-Utilities
# =============================================================================

def find_per_label_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Findet optimalen Threshold pro Label für Multi-Label-Klassifikation.

    Parameter
    ---------
    y_true : np.ndarray
        Wahre Labels (n_samples, n_labels)
    y_proba : np.ndarray
        Vorhergesagte Wahrscheinlichkeiten (n_samples, n_labels)
    label_names : list of str
        Namen der Labels
    thresholds : np.ndarray, optional
        Zu durchsuchende Thresholds (Standard: 0.05 bis 0.95)

    Rückgabe
    --------
    dict
        {label_name: best_threshold}
    """
    from sklearn.metrics import f1_score

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    result = {}
    for j, label in enumerate(label_names):
        best_thr, best_f1 = 0.5, -1.0
        for thr in thresholds:
            pred = (y_proba[:, j] >= thr).astype(int)
            f1 = f1_score(y_true[:, j], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        result[label] = best_thr

    return result


def evaluate_multilabel_with_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Dict[str, float],
    label_names: List[str],
) -> Dict[str, float]:
    """
    Evaluiert Multi-Label-Vorhersagen mit per-Label-Thresholds.

    Rückgabe
    --------
    dict
        {micro_f1, macro_f1, per_label_f1}
    """
    from sklearn.metrics import f1_score

    y_pred = np.zeros_like(y_true)
    for j, label in enumerate(label_names):
        y_pred[:, j] = (y_proba[:, j] >= thresholds.get(label, 0.5)).astype(int)

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label = {
        label: float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))
        for j, label in enumerate(label_names)
    }

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "per_label_f1": per_label,
    }


# =============================================================================
# Ranking-Utilities
# =============================================================================

def mean_ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: List[int],
    k: int = 10,
) -> float:
    """
    Berechnet mittleren NDCG@k über Query-Gruppen.

    Parameter
    ---------
    y_true : np.ndarray
        Wahre Relevanz-Labels
    y_score : np.ndarray
        Vorhergesagte Scores
    groups : list of int
        Größe jeder Query-Gruppe
    k : int
        Cutoff für NDCG-Berechnung

    Rückgabe
    --------
    float
        Mittlerer NDCG@k über alle Gruppen
    """
    import math

    out = []
    start = 0
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    for g in groups:
        if g < 2:
            start += g
            continue

        yt = y_true[start:start + g]
        ys = y_score[start:start + g]
        start += g

        # DCG
        order = np.argsort(-ys)
        yt_sorted = yt[order][:k]

        dcg = 0.0
        for i, rel in enumerate(yt_sorted, start=1):
            dcg += (2 ** rel - 1) / math.log2(i + 1)

        # IDCG
        ideal = np.sort(yt)[-k:][::-1]
        idcg = 0.0
        for i, rel in enumerate(ideal, start=1):
            idcg += (2 ** rel - 1) / math.log2(i + 1)

        out.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(out)) if out else 0.0


def pct_to_relevance(arr: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Konvertiert Perzentil-Werte zu Relevanz-Labels.

    Parameter
    ---------
    arr : np.ndarray
        Perzentil-Werte (0-100)
    n_bins : int
        Anzahl Relevanz-Bins (Standard: 5 -> 0-4)

    Rückgabe
    --------
    np.ndarray
        Relevanz-Labels (0 bis n_bins-1)
    """
    if n_bins == 5:
        # Standard 5-Bin-Relevanz (0=niedrig, 4=hoch)
        return np.select(
            [arr < 50, arr < 80, arr < 95, arr < 99, arr >= 99],
            [0, 1, 2, 3, 4],
        ).astype(int)
    elif n_bins == 32:
        # Feingranulare 32-Bin-Relevanz
        rel = np.floor(arr / (100.0 / 32.0)).astype(int)
        return np.clip(rel, 0, 31).astype(np.int32)
    else:
        # Allgemeiner Fall
        rel = np.floor(arr / (100.0 / n_bins)).astype(int)
        return np.clip(rel, 0, n_bins - 1).astype(int)


# =============================================================================
# Clustering-Suchkonfiguration (Optional)
# =============================================================================

@dataclass
class ClusteringSearchConfig:
    """
    Konfiguration für Clustering-Hyperparameter-Suche.

    Attribute
    ---------
    k_range : tuple
        Bereich der Cluster-Anzahl (min, max)
    pca_dim_range : tuple
        Bereich der PCA-Dimensionen (min, max)
    scale : bool
        Ob Features standardisiert werden sollen
    n_init : int
        Anzahl KMeans-Initialisierungen
    """
    k_range: Tuple[int, int] = (10, 50)
    pca_dim_range: Tuple[int, int] = (8, 32)
    scale: bool = True
    n_init: int = 10

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Schlägt Clustering-Hyperparameter vor."""
        return {
            "k": trial.suggest_int("k", self.k_range[0], self.k_range[1]),
            "pca_dim": trial.suggest_int("pca_dim", self.pca_dim_range[0], self.pca_dim_range[1]),
            "scale": trial.suggest_categorical("scale", [True]) if self.scale else trial.suggest_categorical("scale", [True, False]),
        }

