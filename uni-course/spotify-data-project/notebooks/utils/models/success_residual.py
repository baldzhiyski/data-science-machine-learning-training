from __future__ import annotations
from dataclasses import dataclass
from xgboost import XGBRegressor
from ..data.splits import cohort_time_split
from ..evaluation.metrics import regression_metrics
from ..data.preprocess import TabularPreprocessor

from .tuning_utils import (
    create_optuna_study,
    xgb_device_kwargs,
    suggest_xgb_regression_params,
    collect_tuning_artifacts,
    EARLY_STOPPING_ROUNDS,
)

"""
Task: Erfolgs-Residual innerhalb Kohorte (Regression).

Tuning-Strategie:
-----------------
- Primäre Metrik: MAE (Mean Absolute Error) auf Validierungs-Split
- XGBoost-Objective: reg:absoluteerror (richtet sich am MAE aus)
- Early Stopping: 300 Runden mit n_estimators=20000 Obergrenze
- Stärkere Regularisierung als success_pct (Residual-Targets sind verrauschter)

Warum MAE für Residuals?
- Residual-Vorhersagen zentrieren um 0
- MAE ist robust gegenüber Ausreißern in Residual-Verteilungen
- Negativer R² auf Val/Test signalisiert Overfitting oder Target-Probleme

Hinweis:
Residual-Targets sind oft verrauschter -> stärkere Regularisierung sinnvoll.
Negativer R² auf Val/Test ist ein typisches Symptom für Overfitting oder Target-/Split-Bugs.

Ziel:
Modelliert "Überperformance" relativ zur Kohorte (Residual statt roher Erfolg).
"""


@dataclass
class SuccessResidualTrainer:
    seed: int = 42

    def fit_eval(self, ds, params : dict | None = None):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]
        Xte, yte = ds.X.iloc[idx_te], ds.y.iloc[idx_te]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

        # Stärker regularisiert als success_pct (Residual-Target ist verrauschter)
        if params:
            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                **params
            )
        else:
            model = XGBRegressor(
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                min_child_weight=5.0,
                tree_method="hist",
                random_state=self.seed,
                n_jobs=4,
            )
        model.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

        metrics = {
            "train": regression_metrics(ytr, model.predict(Xtr_p)),
            "val": regression_metrics(yva, model.predict(Xva_p)),
            "test": regression_metrics(yte, model.predict(Xte_p)),
        }
        return model, metrics


    def tune(self, ds, n_trials: int = 50, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Erfolgs-Residual (Regression).

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Minimiere MAE auf Validierungs-Split
        - XGBoost-Objective: reg:absoluteerror (richtet sich am MAE aus)
        - eval_metric: mae (für Early Stopping verwendet)
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Anti-Overfit-Maßnahmen (Stärker als success_pct):
        ------------------------------------------------
        - n_estimators=20000 mit 300 Runden Early Stopping
        - max_depth 3-7 (niedrigere Kapazität für verrauschte Residual-Targets)
        - max_leaves 16-128 (Hist-Regularisierer)
        - Stärkere Regularisierung für Residual-Rauschen empfohlen
        - Subsampling 0.6-0.9 um Varianz zu reduzieren

        Monitoring:
        ----------
        - R² tracken um Overfitting zu erkennen (negativer R² = Problem)
        - Train/Val/Test-Metriken in fit_eval für Diagnostik berichten
        """
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        # Zusätzliche Metriken über Trials tracken
        best_trial_r2 = {"r2": float("-inf"), "trial": -1}

        def objective(trial):
            nonlocal best_trial_r2

            # Suchraum von unified utilities holen
            params = suggest_xgb_regression_params(trial)

            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                objective="reg:absoluteerror",  # Richtet sich am MAE aus
                eval_metric="mae",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **xgb_device_kwargs(device),
                **params,
            )

            model.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

            # Mit bester Iteration vorhersagen wenn Early Stopping ausgelöst
            if getattr(model, "best_iteration", None) is not None:
                pred = model.predict(Xva_p, iteration_range=(0, model.best_iteration + 1))
            else:
                pred = model.predict(Xva_p)

            metrics = regression_metrics(yva, pred)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            # Besten R² als sekundäre Metrik tracken (um Overfitting zu erkennen)
            if r2 > best_trial_r2["r2"]:
                best_trial_r2 = {"r2": r2, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("rmse", metrics["RMSE"])
            trial.set_user_attr("r2", r2)
            trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))

            return float(mae)  # MAE minimieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="minimize",
            seed=self.seed,
            study_name="success_residual_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name="mae",
            device=device,
            best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "best_r2_across_trials": best_trial_r2["r2"],
                "best_trial_rmse": study.best_trial.user_attrs.get("rmse"),
                "best_trial_r2": study.best_trial.user_attrs.get("r2"),
            },
        )

        return result.to_dict()
