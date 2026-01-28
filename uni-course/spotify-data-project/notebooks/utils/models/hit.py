from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from ..data.splits import cohort_time_split
from ..evaluation.metrics import binary_metrics, find_best_threshold_f1
from ..data.preprocess import TabularPreprocessor

from .tuning_utils import (
    create_optuna_study,
    xgb_device_kwargs,
    suggest_xgb_classification_params,
    collect_tuning_artifacts,
    EARLY_STOPPING_ROUNDS,
)

"""
Task: Hit Prediction (Binäre Klassifikation).

Tuning-Strategie:
-----------------
- Primäre Metrik: PR-AUC (Average Precision) auf Validierungs-Split
- Sekundäre Metrik: Bester F1 bei optimalem Threshold (aufgezeichnet, nicht optimiert)
- Early Stopping: 300 Runden mit n_estimators=20000 Obergrenze
- Ungleichgewicht: scale_pos_weight um Klassen-Ratio getuned (0.7x bis 1.5x)
- Kapazität: max_depth 3-7, min_child_weight 3-50 (verhindert kleine Blätter)
- Regularisierung: reg_lambda 1-50, reg_alpha 0.01-2 (sinnvolle Bereiche)
- Threshold: Nach Training auf Validierung optimiert (nicht während Tuning)

Warum PR-AUC?
- ROC-AUC ist unempfindlich gegenüber Klassen-Ungleichgewicht
- PR-AUC fokussiert auf Precision-Recall-Trade-off für Minderheitsklasse
- Reflektiert reale Performance für Hit-Vorhersage besser

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

        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]
        Xte, yte = ds.X.iloc[idx_te], ds.y.iloc[idx_te]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])  # no raw text in your schema
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

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
        clf.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

        proba_va = clf.predict_proba(Xva_p)[:, 1]
        thr, thr_f1 = find_best_threshold_f1(yva, proba_va)

        proba_te = clf.predict_proba(Xte_p)[:, 1]
        m = binary_metrics(yte, proba_te, threshold=thr)
        m["best_threshold"] = float(thr)
        m["best_threshold_f1_on_val"] = float(thr_f1)
        return clf, m, thr

    def tune(self, ds, n_trials: int = 60, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Hit Prediction (Binäre Klassifikation).

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Maximiere PR-AUC (Average Precision) auf Validierung
        - Sekundäre Aufzeichnung: Bester F1 + optimaler Threshold pro Trial
        - Early Stopping: 300 Runden Geduld mit 20k Estimator-Obergrenze
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Ungleichgewichts-Behandlung:
        ---------------------------
        - scale_pos_weight um Klassen-Ratio getuned (keine extremen Bereiche)
        - max_delta_step für numerische Stabilität
        - eval_metric='aucpr' richtet XGBoost am PR-AUC-Ziel aus

        Anti-Overfit-Maßnahmen:
        ----------------------
        - max_depth auf 7 begrenzt (zeitliche Splits brauchen einfachere Modelle)
        - min_child_weight >= 3 (verhindert kleine Blatt-Splits)
        - Sinnvolle Regularisierung (reg_lambda >= 1, reg_alpha >= 0.01)
        - Subsampling 0.6-0.9 (reduziert Varianz)
        """
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)
        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        # Basis-Klassen-Ungleichgewichts-Ratio berechnen
        pos = float(np.sum(ytr == 1))
        neg = float(np.sum(ytr == 0))
        base_spw = (neg / pos) if pos > 0 else 1.0

        # Besten F1/Threshold über Trials tracken (sekundäre Metrik)
        best_trial_f1_info = {"f1": 0.0, "threshold": 0.5, "trial": -1}

        def objective(trial):
            nonlocal best_trial_f1_info

            # Suchraum von unified utilities holen
            params = suggest_xgb_classification_params(trial, base_scale_pos_weight=base_spw)

            clf = XGBClassifier(
                random_state=self.seed,
                n_jobs=4,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric="aucpr",  # Richtet sich am PR-AUC-Ziel aus
                **xgb_device_kwargs(device),
                **params,
            )

            clf.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

            # Mit bester Iteration vorhersagen wenn Early Stopping ausgelöst
            if getattr(clf, "best_iteration", None) is not None:
                proba = clf.predict_proba(Xva_p, iteration_range=(0, clf.best_iteration + 1))[:, 1]
            else:
                proba = clf.predict_proba(Xva_p)[:, 1]

            # Primäre Metrik: PR-AUC
            pr_auc = average_precision_score(yva, proba)

            # Sekundäre Metrik: Bester F1 bei optimalem Threshold (aufgezeichnet, nicht optimiert)
            thr, thr_f1 = find_best_threshold_f1(yva, proba)
            if thr_f1 > best_trial_f1_info["f1"]:
                best_trial_f1_info = {"f1": thr_f1, "threshold": thr, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("best_f1", thr_f1)
            trial.set_user_attr("best_threshold", thr)
            trial.set_user_attr("best_iteration", getattr(clf, "best_iteration", None))

            return pr_auc  # PR-AUC maximieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="maximize",
            seed=self.seed,
            study_name="hit_prediction_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name="pr_auc",
            device=device,
            best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "best_f1_across_trials": best_trial_f1_info["f1"],
                "best_f1_threshold": best_trial_f1_info["threshold"],
                "best_f1_trial": best_trial_f1_info["trial"],
                "class_imbalance_ratio": base_spw,
            },
        )

        return result.to_dict()
