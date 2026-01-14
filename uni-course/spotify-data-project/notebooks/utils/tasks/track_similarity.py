from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


"""
Task: Track Similarity (Embedding / Nearest Neighbors).

Ziel:
Erzeuge kompaktes Embedding (z.B. SVD/Autoencoder) aus Audio-Features
und ermögliche Ähnlichkeitssuche via Cosine Similarity.

Wichtig:
- Track-IDs müssen exakt zum Embedding-Array passen (gleiche Reihenfolge).
- Kein str() Casting auf Index, sonst gibt es Key-Mismatches.
"""


@dataclass
class TrackSimilarityRunner:
    emb_dim: int = 9
    audio_cols: tuple = (
        "acousticness","danceability","energy","instrumentalness",
        "liveness","loudness","speechiness","tempo","valence"
    )

    def fit(self, track_df: pd.DataFrame):
        cols = [c for c in self.audio_cols if c in track_df.columns]
        X = track_df[cols].fillna(0).to_numpy()

        X = StandardScaler().fit_transform(X)
        svd = TruncatedSVD(n_components=min(self.emb_dim, X.shape[1]), random_state=42)
        emb = svd.fit_transform(X)

        artifact = {
            "n_tracks": int(len(track_df)),
            "emb_dim": int(emb.shape[1]),
            "audio_cols_used": cols,
            "example_key": str(track_df.index[0]) if len(track_df.index) else None,
        }
        return {"svd": svd, "embeddings": emb, "cols": cols}, artifact

    @staticmethod
    def get_similar(track_key: str, track_index: pd.Index, embeddings: np.ndarray, top_k: int = 10):
        if track_key not in track_index:
            return []
        i = track_index.get_loc(track_key)
        v = embeddings[i]
        # cosine sim
        denom = (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(v) + 1e-9))
        sims = (embeddings @ v) / (denom + 1e-9)
        order = np.argsort(-sims)
        out = []
        for j in order[1:top_k+1]:
            out.append((str(track_index[j]), float(sims[j])))
        return out
