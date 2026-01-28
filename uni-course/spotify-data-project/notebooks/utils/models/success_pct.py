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
Task: Erfolgs-Perzentil-Vorhersage (Regression).

Tuning-Strategie:
-----------------
- Primäre Metrik: MAE (Mean Absolute Error) auf Validierungs-Split
- XGBoost-Objective: reg:absoluteerror (richtet sich am MAE aus)
- Early Stopping: 300 Runden mit n_estimators=20000 Obergrenze
- Kapazität: max_depth 3-7, max_leaves 16-128 (beschränkt)
- Regularisierung: reg_lambda 1-50, reg_alpha 0.01-2 (sinnvolle Bereiche)
- Subsampling: 0.6-0.9 für Varianz-Reduktion

Warum MAE?
- Perzentil-Vorhersagen haben intuitive Interpretation
- MAE ist robust gegenüber Ausreißern in schiefen Verteilungen
- RMSE als sekundäre Metrik aufgezeichnet

Ziel:
Vorhersage des Erfolgperzentils (0-100) eines Tracks innerhalb seiner Kohorte.
Hinweise:
- Kohortenbasierte Zeit-Splits sind wichtig, um Daten-Leakage zu vermeiden.
- Erfolgperzentile sind oft schief verteilt; geeignete Metriken wählen.
"""

@dataclass
class SuccessPctTrainer:
    seed: int = 42

    def fit_eval(self, ds,params : dict | None =None):
        idx_tr, idx_va, idx_te = cohort_time_split(ds.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]
        Xte, yte = ds.X.iloc[idx_te], ds.y.iloc[idx_te]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

        if params:
            model = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                **params
            )
        else:
            model = XGBRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=self.seed,
                n_jobs=4,
            )
        model.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

        pred_te = model.predict(Xte_p)
        m = regression_metrics(yte, pred_te)
        m.update({
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "n_test": int(len(yte)),
            "label_range_expected": [0, 100],
        })
        return model, m

    def tune(self, ds, n_trials: int = 40, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Erfolgs-Perzentil (Regression).

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Minimiere MAE auf Validierungs-Split
        - XGBoost-Objective: reg:absoluteerror (richtet sich am MAE aus)
        - eval_metric: mae (für Early Stopping verwendet)
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Anti-Overfit-Maßnahmen:
        ----------------------
        - n_estimators=20000 mit 300 Runden Early Stopping
        - max_depth 3-7 (niedrigere Kapazität für Zeit-Split-Generalisierung)
        - max_leaves 16-128 (Hist-Regularisierer)
        - Sinnvolle Regularisierung (reg_lambda >= 1, reg_alpha >= 0.01)
        - Subsampling 0.6-0.9 um Varianz zu reduzieren
        """
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)
        Xtr, ytr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr]
        Xva, yva = ds.X.iloc[idx_va], ds.y.iloc[idx_va]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        # Zusätzliche Metriken über Trials tracken
        best_trial_rmse = {"rmse": float("inf"), "trial": -1}

        def objective(trial):
            nonlocal best_trial_rmse

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
            rmse = metrics["RMSE"]

            # Besten RMSE als sekundäre Metrik tracken
            if rmse < best_trial_rmse["rmse"]:
                best_trial_rmse = {"rmse": rmse, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("r2", metrics["R2"])
            trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))

            return float(mae)  # MAE minimieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="minimize",
            seed=self.seed,
            study_name="success_pct_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name="mae",
            device=device,
            best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "best_rmse_across_trials": best_trial_rmse["rmse"],
                "best_trial_r2": study.best_trial.user_attrs.get("r2"),
            },
        )

        return result.to_dict()

