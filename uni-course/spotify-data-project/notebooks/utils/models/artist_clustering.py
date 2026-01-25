from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass
class ArtistClusteringRunner:
    k: int = 30
    seed: int = 42
    scale: bool = True
    pca_dim: int = 16

    def run(self, artist_df: pd.DataFrame):

        X_df = (
            artist_df
            .select_dtypes(include=["number", "bool"])
            .copy()
        )

        # make sure we have something to cluster
        if X_df.shape[1] == 0:
            raise ValueError("No numeric/bool features found for clustering.")

        # fill missing + cast to float
        X_df = X_df.fillna(0).astype(float)

        # ---- scaling (recommended for KMeans + PCA)
        scaler = None
        X_used = X_df.to_numpy()

        if self.scale:
            scaler = StandardScaler()
            X_used = scaler.fit_transform(X_used)

        # ---- PCA
        n_components = min(self.pca_dim, X_used.shape[1])
        if n_components < 1:
            raise ValueError("PCA components ended up < 1. Check your feature matrix.")

        pca = PCA(n_components=n_components, random_state=self.seed)
        Xp = pca.fit_transform(X_used)

        # ---- KMeans
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
        extra = {"X_used": X_used, "labels": labels, "X_df_cols": list(X_df.columns)}

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