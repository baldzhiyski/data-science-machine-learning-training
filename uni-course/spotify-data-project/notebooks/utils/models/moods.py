from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from ..data.splits import cohort_time_split
from ..data.preprocess import TabularPreprocessor

from .tuning_utils import (
    create_optuna_study,
    suggest_sgd_multilabel_params,
    collect_tuning_artifacts,
)

"""
Task: Mood/Genre Multi-Label-Klassifikation.

Tuning-Strategie:
-----------------
- Primäre Metrik: micro-F1 auf Validierungs-Split (Gesamt-Sample-Performance)
- Sekundäre Metrik: macro-F1 (überwacht per-Label-Balance)
- Globales Threshold-Tuning (großer Einfluss auf micro-F1)
- Per-Label-Thresholds nach Tuning in fit_eval berechnet

Warum micro-F1?
- Optimiert Vorhersagen auf Sample-Ebene insgesamt
- Stabiler für unbalancierte Label-Verteilungen
- macro-F1 getrackt um Label-Kollaps zu erkennen

Threshold-Auswahl:
-----------------
- Einzelner globaler Threshold während Optuna-Suche getuned (vereinfacht Optimierung)
- Per-Label-Thresholds während Evaluation berechnet (bessere finale Performance)
- Vermeidet Overfitting auf micro-F1 während macro-F1 kollabiert

Ziel:
Vorhersage mehrerer Stimmungs- oder Genre-Labels pro Track.
Hinweise:
- Multi-Label Klassifikation mit unabhängigen Labels.
- Per-Label Threshold-Optimierung zur Verbesserung der F1-Metriken.
"""

@dataclass
class MoodTrainer:
    seed: int = 42

    def fit_eval(self, ds, params : dict | None = None):
        # keep it simple: random split (you can swap to cohort split later if you want)
        Xtr, Xte, Ytr, Yte = train_test_split(
            ds.X, ds.y, test_size=0.20, random_state=self.seed
        )
        pre = TabularPreprocessor(model_kind="linear", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xte_p = ct.transform(Xte)

        if params:
            base = SGDClassifier(
                random_state=self.seed,
                max_iter=2000,
                class_weight="balanced",
                **params)
        else:
            base = SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=2000,
                class_weight="balanced",
                random_state=self.seed,
            )
        clf = OneVsRestClassifier(base)
        clf.fit(Xtr_p, Ytr)

        # probabilities
        P = clf.predict_proba(Xte_p)

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
        Hyperparameter-Tuning für Mood Multi-Label-Klassifikation.

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Maximiere micro-F1 auf Validierungs-Split
        - Sekundäres Tracking: macro-F1 um Label-Kollaps zu erkennen
        - Globales Threshold-Tuning (0.1-0.9) für Einfachheit und Stabilität
        - Reproduzierbar: Fester TPE-Sampler-Seed

        SGDClassifier-Tuning:
        --------------------
        - alpha: Regularisierungsstärke (1e-6 bis 1e-2, log-Skala)
        - loss: log_loss oder modified_huber (wahrscheinlichkeitsähnliche Scores)
        - penalty: l2, l1, oder elasticnet mit l1_ratio-Tuning

        Threshold-Behandlung:
        --------------------
        - Einzelner globaler Threshold während Tuning (effiziente Suche)
        - Per-Label-Thresholds in fit_eval berechnet (bessere finale Performance)
        - decision_function + sigmoid für robuste "wahrscheinlichkeitsähnliche" Scores

        Anti-Overfit-Maßnahmen:
        ----------------------
        - Early Stopping innerhalb SGD (validation_fraction=0.1)
        - n_iter_no_change=10 Geduld
        - macro-F1 überwachen um micro-F1-Gaming auf Label-Kosten zu erkennen
        """
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, Ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, Yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        pre = TabularPreprocessor(model_kind="linear", text_cols=[])
        ct = pre.build(Xtr)
        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        label_names = list(ds.y.columns)

        def sigmoid(z):
            z = np.clip(z, -20, 20)  # Numerische Stabilität
            return 1.0 / (1.0 + np.exp(-z))

        def get_scores_ovr(model, X):
            """
            Gibt (n_samples, n_labels) Score-Matrix in [0,1] zurück.
            Funktioniert auch wenn predict_proba nicht verfügbar ist.
            """
            scores = []
            for est in model.estimators_:
                if hasattr(est, "predict_proba"):
                    s = est.predict_proba(X)[:, 1]
                else:
                    s = sigmoid(est.decision_function(X))
                scores.append(s)
            return np.column_stack(scores)

        # macro-F1 tracken um Label-Kollaps zu erkennen
        best_macro_f1_info = {"macro_f1": 0.0, "trial": -1}

        def objective(trial):
            nonlocal best_macro_f1_info

            # Suchraum von unified utilities holen
            sgd_params = suggest_sgd_multilabel_params(trial)

            # Globalen Threshold tunen (großer Gewinn für micro-F1)
            thr = trial.suggest_float("threshold", 0.1, 0.9)

            base = SGDClassifier(
                random_state=self.seed,
                alpha=sgd_params["alpha"],
                loss=sgd_params["loss"],
                penalty=sgd_params["penalty"],
                l1_ratio=sgd_params["l1_ratio"] if sgd_params["penalty"] == "elasticnet" else 0.15,
                max_iter=4000,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            )

            model = OneVsRestClassifier(base, n_jobs=4)
            model.fit(Xtr_p, Ytr)

            proba_like = get_scores_ovr(model, Xva_p)
            pred = (proba_like >= thr).astype(int)

            micro_f1 = f1_score(Yva.values, pred, average="micro", zero_division=0)
            macro_f1 = f1_score(Yva.values, pred, average="macro", zero_division=0)

            # macro-F1 tracken um Label-Kollaps zu erkennen
            if macro_f1 > best_macro_f1_info["macro_f1"]:
                best_macro_f1_info = {"macro_f1": macro_f1, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("macro_f1", macro_f1)
            trial.set_user_attr("threshold", thr)
            trial.set_user_attr("micro_macro_ratio", micro_f1 / max(macro_f1, 1e-6))

            return float(micro_f1)  # micro-F1 maximieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="maximize",
            seed=self.seed,
            study_name="mood_multilabel_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name="micro_f1",
            device="cpu",
            extra_metrics={
                "best_macro_f1_across_trials": best_macro_f1_info["macro_f1"],
                "best_trial_macro_f1": study.best_trial.user_attrs.get("macro_f1"),
                "best_threshold": study.best_trial.user_attrs.get("threshold"),
                "micro_macro_ratio": study.best_trial.user_attrs.get("micro_macro_ratio"),
                "n_labels": len(label_names),
            },
        )

        return result.to_dict()
