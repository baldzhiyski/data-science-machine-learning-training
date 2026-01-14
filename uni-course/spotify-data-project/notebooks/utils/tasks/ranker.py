from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
from ..splits import cohort_time_split
import optuna


def _to_relevance_0_31(y_pct):
    # simple mapping: percentile -> relevance bin 0..31
    y = np.asarray(y_pct)
    rel = np.floor(y / (100.0 / 32.0)).astype(int)
    return np.clip(rel, 0, 31).astype(np.int32)


"""
Task: Success Percentile Ranking.
Ziel:
Modelliert die Rangordnung von Tracks innerhalb ihrer Kohorte basierend auf Erfolgpercentilen.
Hinweise:
- Nutzt Learning-to-Rank-Methoden (z.B. LambdaMART) für
    kohortenbasierte Rangvorhersagen.
- Kohortenbasierte Zeit-Splits sind wichtig, um Daten-Leakage zu vermeiden.
- Relevanz-Binning (0-31) wird verwendet, um kontinuierliche Percentile in diskrete Relevanzstufen umzuwandeln.
"""

@dataclass
class RankerTrainer:
    seed: int = 42
    k_eval: int = 10

    def fit_eval(self, ds_success_pct):
        y_rel = _to_relevance_0_31(ds_success_pct.y)

        idx_tr, idx_va, idx_te = cohort_time_split(ds_success_pct.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr, Xva, Xte = ds_success_pct.X[idx_tr], ds_success_pct.X[idx_va], ds_success_pct.X[idx_te]
        ytr, yva, yte = y_rel[idx_tr], y_rel[idx_va], y_rel[idx_te]

        # groups = per cohort counts
        def group_sizes(meta):
            return meta.groupby("cohort_ym").size().astype(int).tolist()

        gtr = group_sizes(ds_success_pct.meta[idx_tr])
        gva = group_sizes(ds_success_pct.meta[idx_va])
        gte = group_sizes(ds_success_pct.meta[idx_te])

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

        model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva], verbose=False)

        # NDCG on test computed per cohort, then averaged
        meta_te = ds_success_pct.meta[idx_te].reset_index(drop=True)
        scores = model.predict(Xte)

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
        Optimiert mean NDCG@k auf dem Validierungs-Split.

        Fixes included:
        - Convert float percentiles -> integer relevance grades (0..4) (rank:ndcg requires int labels, and <=31 for exp gain)
        - Ensure contiguous groups by sorting by cohort_ym
        - Early stopping + use best_iteration for prediction
        """

        import numpy as np
        import optuna
        from xgboost import XGBRanker

        def _xgb_device_kwargs(dev: str | None):
            dev = (dev or "cpu").lower().strip()
            if dev == "gpu":
                dev = "cuda"
            if dev.startswith("cuda"):
                return {"tree_method": "hist", "device": dev}  # "cuda" or "cuda:0"
            return {"tree_method": "hist", "device": "cpu"}

        def build_group_sizes(meta_df):
            # meta_df MUST be sorted so that cohort_ym is contiguous
            return meta_df.groupby("cohort_ym", sort=False).size().to_list()

        def pct_to_rel_0_4(arr: np.ndarray) -> np.ndarray:
            """
            Convert success percentile [0..100] floats into small integer relevance grades.
            Keeps max rel <= 4 (safe for rank:ndcg exponential gain).
            """
            return np.select(
                [arr < 50, arr < 80, arr < 95, arr < 99, arr >= 99],
                [0, 1, 2, 3, 4],
            ).astype(int)

        def mean_ndcg_at_k(y_true, y_score, groups, k=10):
            # simple NDCG@k per group
            import math

            out = []
            start = 0
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            for g in groups:
                yt = y_true[start:start + g]
                ys = y_score[start:start + g]
                start += g

                order = np.argsort(-ys)
                yt_sorted = yt[order][:k]

                # DCG
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

        # ---- Split (time-based by cohort) ----
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)

        Xtr, ytr, mtr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr], ds.meta.iloc[idx_tr]
        Xva, yva, mva = ds.X.iloc[idx_va], ds.y.iloc[idx_va], ds.meta.iloc[idx_va]

        # ---- Ensure groups are contiguous: sort within each split ----
        tr_order = mtr.sort_values("cohort_ym").index
        va_order = mva.sort_values("cohort_ym").index

        Xtr, ytr, mtr = ds.X.loc[tr_order], ds.y.loc[tr_order], ds.meta.loc[tr_order]
        Xva, yva, mva = ds.X.loc[va_order], ds.y.loc[va_order], ds.meta.loc[va_order]

        # ---- Build group sizes (cohort-based grouping) ----
        group_tr = build_group_sizes(mtr)
        group_va = build_group_sizes(mva)

        # Optional safety checks (helpful during development)
        if sum(group_tr) != len(Xtr):
            raise ValueError(f"group_tr sum {sum(group_tr)} != len(Xtr) {len(Xtr)}")
        if sum(group_va) != len(Xva):
            raise ValueError(f"group_va sum {sum(group_va)} != len(Xva) {len(Xva)}")

        # ---- Convert float percentiles -> integer relevance grades ----
        # ds.y is a Series of float percentiles [0..100]
        ytr_rel = pct_to_rel_0_4(ytr.to_numpy())
        yva_rel = pct_to_rel_0_4(yva.to_numpy())

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 6000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 80.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            ranker = XGBRanker(
                objective="rank:ndcg",
                random_state=self.seed,
                n_jobs=4,
                early_stopping_rounds=100,
                eval_metric=f"ndcg@{k}",
                **_xgb_device_kwargs(device),
                **params,
            )

            ranker.fit(
                Xtr, ytr_rel,
                group=group_tr,
                eval_set=[(Xva, yva_rel)],
                eval_group=[group_va],
                verbose=False,
            )

            # Use best_iteration due to early stopping
            if getattr(ranker, "best_iteration", None) is not None:
                y_score = ranker.predict(Xva, iteration_range=(0, ranker.best_iteration + 1))
            else:
                y_score = ranker.predict(Xva)

            return mean_ndcg_at_k(yva_rel, y_score, group_va, k=k)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": dict(study.best_params),
            "best_val_mean_ndcg@k": float(study.best_value),
            "k": k,
            "device": device,
        }