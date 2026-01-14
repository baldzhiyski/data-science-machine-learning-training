from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

"""
Task: Artist Clustering (Unsupervised).

Ziel:
Künstler anhand numerischer Features clustern (PCA -> KMeans).

Output:
- Modelle (Scaler/PCA/KMeans)
- Artefakt-Dict für Report
- optional extra (X_used, labels) für Notebook-Plots

Hinweis:
Clustering hat kein klassisches "Ground Truth"-Metric; Validierung erfolgt oft qualitativ/visuell.
"""


@dataclass
class ArtistClusteringRunner:
    k: int = 30
    seed: int = 42
    scale: bool = True
    pca_dim: int = 16

    def run(self, artist_df: pd.DataFrame):
        X_df = artist_df.select_dtypes(include=["number", "bool"]).fillna(0).astype(float)

        # Optional scaling (recommended)
        scaler = None
        X_used = X_df.to_numpy()
        if self.scale:
            scaler = StandardScaler()
            X_used = scaler.fit_transform(X_used)

        pca = PCA(n_components=min(self.pca_dim, X_used.shape[1]), random_state=self.seed)
        Xp = pca.fit_transform(X_used)

        km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        labels = km.fit_predict(Xp)

        artifact = {
            "k": int(self.k),
            "n_artists": int(len(artist_df)),
            "n_features_used": int(X_df.shape[1]),
            "scaled": bool(self.scale),
            "pca_components": int(pca.n_components_),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "tsne_sample_n": 4000,
        }

        models = {"scaler": scaler, "pca": pca, "kmeans": km}
        extra = {"X_used": X_used, "labels": labels}  # <-- for plotting

        return models, artifact, extra

    def plot_pca2(self, X_used, labels, title="Artist Clusters (PCA 2D)"):
        pca2 = PCA(n_components=2, random_state=self.seed)
        X_pca2 = pca2.fit_transform(X_used)

        plt.figure(figsize=(9, 6))
        plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels, s=12, alpha=0.7)
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
