from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
from ..data.splits import cohort_time_split
from ..data.preprocess import TabularPreprocessor

from .tuning_utils import (
    create_optuna_study,
    xgb_device_kwargs,
    suggest_xgb_ranker_params,
    collect_tuning_artifacts,
    mean_ndcg_at_k,
    pct_to_relevance,
    EARLY_STOPPING_ROUNDS,
)


def _to_relevance_0_31(y_pct):
    # Einfache Abbildung: Perzentil -> Relevanz-Bin 0..31
    y = np.asarray(y_pct)
    rel = np.floor(y / (100.0 / 32.0)).astype(int)
    return np.clip(rel, 0, 31).astype(np.int32)


"""
Task: Erfolgs-Perzentil-Ranking (Learning-to-Rank).

Tuning-Strategie:
-----------------
- Primäre Metrik: mittlerer NDCG@k auf Validierungs-Split
- XGBoost-Objective: rank:ndcg (LambdaMART-ähnlich)
- Early Stopping: 300 Runden mit n_estimators=20000 Obergrenze
- Query-Gruppen: Kohortenbasiert (Tracks derselben Kohorte konkurrieren)

Warum NDCG@k?
- Standard-Ranking-Metrik, die Position und Relevanz berücksichtigt
- Gewichtet niedrigere Positionen logarithmisch ab
- k kontrolliert Tiefe der Ranking-Qualitäts-Optimierung

Gruppen-Behandlung:
------------------
- Gruppen definiert durch cohort_ym (im selben Zeitraum veröffentlicht)
- Gruppen müssen zusammenhängend sein (nach Kohorte sortiert vor Training)
- Gruppengrößen werden validiert um Datenlänge zu entsprechen

Relevanz-Binning:
----------------
- Perzentile (0-100) werden auf Relevanz-Labels (0-4) abgebildet
- 5-Bin-Schema: <50=0, <80=1, <95=2, <99=3, >=99=4
- Erfasst Erfolgsverteilung (Top 1% bekommt höchste Relevanz)

Ziel:
Modelliert die Rangordnung von Tracks innerhalb ihrer Kohorte basierend auf Erfolgperzentilen.
Hinweise:
- Nutzt Learning-to-Rank-Methoden (z.B. LambdaMART) für kohortenbasierte Rangvorhersagen.
- Kohortenbasierte Zeit-Splits sind wichtig, um Daten-Leakage zu vermeiden.
- Relevanz-Binning wird verwendet, um kontinuierliche Perzentile in diskrete Relevanzstufen umzuwandeln.
"""

@dataclass
class RankerTrainer:
    seed: int = 42
    k_eval: int = 10

    def fit_eval(self, ds_success_pct,params : dict | None = None):
        y_rel = _to_relevance_0_31(ds_success_pct.y)

        idx_tr, idx_va, idx_te = cohort_time_split(ds_success_pct.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr = ds_success_pct.X.iloc[idx_tr]
        Xva = ds_success_pct.X.iloc[idx_va]
        Xte = ds_success_pct.X.iloc[idx_te]
        ytr, yva, yte = y_rel[idx_tr], y_rel[idx_va], y_rel[idx_te]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

        # groups = per cohort counts
        def group_sizes(meta):
            return meta.groupby("cohort_ym").size().astype(int).tolist()

        gtr = group_sizes(ds_success_pct.meta[idx_tr])
        gva = group_sizes(ds_success_pct.meta[idx_va])
        gte = group_sizes(ds_success_pct.meta[idx_te])

        if params:
            model = XGBRanker(
                objective="rank:ndcg",
                random_state=self.seed,
                n_jobs=4,
                **params
            )
        else:

            model = XGBRanker(
                objective="rank:ndcg",
                learning_rate=0.05,
                max_depth=6,
                n_estimators=800,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.seed,
                tree_method="hist",
                n_jobs=4,
            )

        model.fit(Xtr_p, ytr, group=gtr, eval_set=[(Xva_p, yva)], eval_group=[gva], verbose=False)

        # NDCG on test computed per cohort, then averaged
        meta_te = ds_success_pct.meta[idx_te].reset_index(drop=True)
        scores = model.predict(Xte_p)

        ndcgs = []
        for cohort, idxs in meta_te.groupby("cohort_ym").groups.items():
            idxs = np.array(list(idxs))
            y_true = yte[idxs].reshape(1, -1)
            y_score = scores[idxs].reshape(1, -1)
            if y_true.shape[1] >= 2:
                ndcgs.append(ndcg_score(y_true, y_score, k=min(self.k_eval, y_true.shape[1])))
        return model, {"mean_ndcg@k": float(np.mean(ndcgs)) if ndcgs else float("nan"), "k": int(self.k_eval)}

    def tune(self, ds, n_trials: int = 30, device: str = "cpu", k: int = 10):
        """
        Hyperparameter-Tuning für Ranking-Modell (XGBRanker).

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Maximiere mittleren NDCG@k auf Validierungs-Split
        - XGBoost-Objective: rank:ndcg (LambdaMART-ähnlich)
        - eval_metric: ndcg@k (für Early Stopping verwendet)
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Gruppen-Behandlung:
        ------------------
        - Gruppen definiert durch cohort_ym (zusammenhängend nach Sortierung)
        - Gruppengrößen vor Training validiert
        - Per-Gruppen NDCG@k berechnet, dann gemittelt

        Anti-Overfit-Maßnahmen:
        ----------------------
        - n_estimators=20000 mit 300 Runden Early Stopping
        - max_depth 3-8 (leicht höhere Kapazität für Ranking erlaubt)
        - max_leaves 16-256 (Hist-Regularisierer)
        - Sinnvolle Regularisierung (reg_lambda >= 1)
        - Subsampling 0.6-0.9 für Varianz-Reduktion
        """
        def build_group_sizes(meta_df):
            return meta_df.groupby("cohort_ym", sort=False).size().to_list()

        # ---- Split (zeitbasiert nach Kohorte) ----
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)

        Xtr, ytr, mtr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr], ds.meta.iloc[idx_tr]
        Xva, yva, mva = ds.X.iloc[idx_va], ds.y.iloc[idx_va], ds.meta.iloc[idx_va]

        # ---- Gruppen zusammenhängend machen (nach Kohorte sortieren) ----
        tr_order = mtr.sort_values("cohort_ym").index
        va_order = mva.sort_values("cohort_ym").index

        Xtr, ytr, mtr = ds.X.loc[tr_order], ds.y.loc[tr_order], ds.meta.loc[tr_order]
        Xva, yva, mva = ds.X.loc[va_order], ds.y.loc[va_order], ds.meta.loc[va_order]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        group_tr = build_group_sizes(mtr)
        group_va = build_group_sizes(mva)

        # Gruppengrößen validieren
        if sum(group_tr) != len(Xtr):
            raise ValueError(f"group_tr sum {sum(group_tr)} != len(Xtr) {len(Xtr)}")
        if sum(group_va) != len(Xva):
            raise ValueError(f"group_va sum {sum(group_va)} != len(Xva) {len(Xva)}")

        # Perzentile zu Relevanz-Labels konvertieren mit shared Utility
        ytr_rel = pct_to_relevance(ytr.to_numpy(), n_bins=5)
        yva_rel = pct_to_relevance(yva.to_numpy(), n_bins=5)

        # Zusätzliche Metriken über Trials tracken
        best_trial_info = {"ndcg": 0.0, "best_iteration": None, "trial": -1}

        def objective(trial):
            nonlocal best_trial_info

            # Suchraum von unified utilities holen
            params = suggest_xgb_ranker_params(trial)

            ranker = XGBRanker(
                objective="rank:ndcg",
                random_state=self.seed,
                n_jobs=4,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric=f"ndcg@{k}",
                **xgb_device_kwargs(device),
                **params,
            )

            ranker.fit(
                Xtr_p, ytr_rel,
                group=group_tr,
                eval_set=[(Xva_p, yva_rel)],
                eval_group=[group_va],
                verbose=False,
            )

            # Mit bester Iteration vorhersagen wenn Early Stopping ausgelöst
            best_iter = getattr(ranker, "best_iteration", None)
            if best_iter is not None:
                y_score = ranker.predict(Xva_p, iteration_range=(0, best_iter + 1))
            else:
                y_score = ranker.predict(Xva_p)

            # Mittleren NDCG@k über Gruppen berechnen mit shared Utility
            ndcg = mean_ndcg_at_k(yva_rel, y_score, group_va, k=k)

            # Bestes Ergebnis tracken
            if ndcg > best_trial_info["ndcg"]:
                best_trial_info = {"ndcg": ndcg, "best_iteration": best_iter, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("best_iteration", best_iter)
            trial.set_user_attr("n_groups_val", len(group_va))

            return ndcg  # NDCG@k maximieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="maximize",
            seed=self.seed,
            study_name="ranker_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name=f"ndcg@{k}",
            device=device,
            best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "k": k,
                "n_groups_train": len(group_tr),
                "n_groups_val": len(group_va),
                "relevance_bins": 5,
            },
        )

        return result.to_dict()
