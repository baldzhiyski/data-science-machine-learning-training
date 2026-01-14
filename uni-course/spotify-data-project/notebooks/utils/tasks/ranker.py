from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
from ..splits import cohort_time_split


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
