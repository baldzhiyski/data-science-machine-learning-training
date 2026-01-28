from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

from .tuning_utils import (
    create_optuna_study,
    collect_tuning_artifacts,
)

"""
Task: Artist-Clustering (Unüberwacht).

Tuning-Strategie (Optional):
----------------------------
- Primäre Metrik: Silhouette-Score (höher = bessere Cluster-Trennung)
- Sekundäre Metrik: Davies-Bouldin-Index (niedriger = besser)
- Suchraum: k (Cluster-Anzahl), pca_dim, Skalierung

Warum Silhouette-Score?
- Misst sowohl Kohäsion als auch Separation
- Funktioniert ohne Ground-Truth-Labels
- Interpretierbarer Bereich [-1, 1]

Monitoring:
----------
- PCA-Varianz-Ratio (sollte angemessen sein, z.B. >0.7)
- Davies-Bouldin-Index (sollte mit besserem Clustering sinken)
- Cluster-Größen-Verteilung (degenerierte Lösungen erkennen)

Ziel:
Gruppierung von Künstlern basierend auf numerischen Features.
"""


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

        # Add clustering quality metrics
        if len(np.unique(labels)) > 1:
            artifact["silhouette_score"] = float(silhouette_score(Xp, labels))
            artifact["davies_bouldin_score"] = float(davies_bouldin_score(Xp, labels))

        models = {"scaler": scaler, "pca": pca, "kmeans": km}
        extra = {"X_used": X_used, "labels": labels, "X_df_cols": list(X_df.columns)}

        return models, artifact, extra

    def tune(self, artist_df: pd.DataFrame, n_trials: int = 30,
             k_range: tuple = (10, 50), pca_dim_range: tuple = (8, 32)):
        """
        Hyperparameter-Tuning für Artist-Clustering.

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Maximiere Silhouette-Score
        - Sekundäres Tracking: Davies-Bouldin-Index, PCA-Varianz
        - Suchraum: k (Cluster-Anzahl), pca_dim
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Suchraum:
        ---------
        - k: Anzahl Cluster (Standard: 10-50)
        - pca_dim: Anzahl PCA-Komponenten (Standard: 8-32)
        - Skalierung: Immer True (erforderlich für KMeans + PCA)

        Monitoring:
        ----------
        - Silhouette-Score sollte > 0 sein (positiv = sinnvolle Cluster)
        - Davies-Bouldin sollte niedrig sein (< 2 ist anständig)
        - PCA-Varianz > 0.7 empfohlen für sinnvolle Reduktion
        """
        # Feature-Matrix vorbereiten
        X_df = (
            artist_df
            .select_dtypes(include=["number", "bool"])
            .copy()
        )

        if X_df.shape[1] == 0:
            raise ValueError("Keine numerischen/bool Features für Clustering gefunden.")

        X_df = X_df.fillna(0).astype(float)

        # Skalieren (immer empfohlen)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df.to_numpy())

        max_pca_dim = min(pca_dim_range[1], X_scaled.shape[1])

        # Besten Davies-Bouldin als sekundäre Metrik tracken
        best_db_info = {"db": float("inf"), "trial": -1}

        def objective(trial):
            nonlocal best_db_info

            k = trial.suggest_int("k", k_range[0], k_range[1])
            pca_dim = trial.suggest_int("pca_dim", pca_dim_range[0], max_pca_dim)

            # PCA
            pca = PCA(n_components=pca_dim, random_state=self.seed)
            Xp = pca.fit_transform(X_scaled)

            # KMeans
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(Xp)

            # Auf degenerierte Lösungen prüfen
            n_unique = len(np.unique(labels))
            if n_unique < 2:
                return -1.0  # Ungültiges Clustering

            # Metriken berechnen
            sil = silhouette_score(Xp, labels)
            db = davies_bouldin_score(Xp, labels)
            pca_var = pca.explained_variance_ratio_.sum()

            # Besten DB tracken
            if db < best_db_info["db"]:
                best_db_info = {"db": db, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("davies_bouldin", db)
            trial.set_user_attr("pca_variance", pca_var)
            trial.set_user_attr("n_unique_clusters", n_unique)
            trial.set_user_attr("inertia", km.inertia_)

            return sil  # Silhouette maximieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="maximize",
            seed=self.seed,
            study_name="artist_clustering_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name="silhouette_score",
            device="cpu",
            extra_metrics={
                "best_k": study.best_params.get("k"),
                "best_pca_dim": study.best_params.get("pca_dim"),
                "best_davies_bouldin": best_db_info["db"],
                "best_trial_pca_variance": study.best_trial.user_attrs.get("pca_variance"),
                "best_trial_inertia": study.best_trial.user_attrs.get("inertia"),
                "n_artists": len(artist_df),
                "n_features": X_df.shape[1],
            },
        )

        return result.to_dict()

    def plot_pca2(self, X_used, labels, title="Artist Clusters (PCA 2D)"):
        pca2 = PCA(n_components=2, random_state=self.seed)
        X_pca2 = pca2.fit_transform(X_used)

        plt.figure(figsize=(9, 6))
        plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels, s=12, alpha=0.7)
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()