from sklearn.cluster import KMeans


def kmeans_compat(n_clusters: int, random_state: int) -> KMeans:
    """
    Zweck:
    - Erstellt ein KMeans-Objekt kompatibel mit verschiedenen scikit-learn Versionen.

    Problem:
    - Neuere Versionen erlauben n_init="auto".
    - Ältere Versionen erwarten eine Zahl (z.B. 10).

    Lösung:
    - Try/Except und fallback.

    Parameter:
    - n_clusters: Anzahl Cluster
    - random_state: sorgt für reproduzierbare Cluster-Ergebnisse
    """
    try:
        return KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    except TypeError:
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)