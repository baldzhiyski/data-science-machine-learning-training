from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


@dataclass
class TrackSimilarityRunner:
    emb_dim: int = 9
    seed: int = 42
    audio_cols: tuple = (
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "loudness", "speechiness", "tempo", "valence"
    )

    def fit(self, track_df: pd.DataFrame):
        cols = [c for c in self.audio_cols if c in track_df.columns]
        if not cols:
            raise ValueError("No audio columns found in track_df for similarity embedding.")

        X = track_df[cols].fillna(0).to_numpy(dtype=float)

        # ✅ keep scaler so embeddings are reproducible for new data
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        svd = TruncatedSVD(n_components=min(self.emb_dim, Xs.shape[1]), random_state=self.seed)
        emb = svd.fit_transform(Xs)

        artifact = {
            "n_tracks": int(len(track_df)),
            "emb_dim": int(emb.shape[1]),
            "audio_cols_used": cols,
            "index_dtype": str(track_df.index.dtype),
            "example_key": track_df.index[0] if len(track_df.index) else None,  # no str() casting
        }

        #  return scaler too
        return {"scaler": scaler, "svd": svd, "embeddings": emb, "cols": cols}, artifact

    @staticmethod
    def get_similar(track_key, track_index: pd.Index, embeddings: np.ndarray, top_k: int = 10):
        # track_key type should match index type (no forced str)
        if track_key not in track_index:
            return []

        i = track_index.get_loc(track_key)
        v = embeddings[i]

        # cosine similarity
        v_norm = np.linalg.norm(v) + 1e-9
        E_norm = np.linalg.norm(embeddings, axis=1) + 1e-9
        sims = (embeddings @ v) / (E_norm * v_norm)

        order = np.argsort(-sims)

        out = []
        for j in order[1: top_k + 1]:
            out.append((track_index[j], float(sims[j])))
        return out