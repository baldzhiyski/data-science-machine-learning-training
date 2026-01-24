import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def onehot_encoder_compat() -> OneHotEncoder:
    """
    Zweck:
    - Erstellt einen OneHotEncoder kompatibel mit verschiedenen scikit-learn Versionen.

    Problem:
    - In neueren sklearn-Versionen heißt der Parameter 'sparse_output'.
    - In älteren Versionen heißt er 'sparse'.

    Lösung:
    - Wir versuchen zuerst die neue Signatur.
    - Falls TypeError: fallback auf die alte Signatur.

    Zusätzlich:
    - handle_unknown="ignore" verhindert Fehler bei neuen/unbekannten Kategorien im Scoring.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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

def sklearn_sanitize_df(X):
    """
        Macht ein pandas-DataFrame sklearn-kompatibel (v. a. für ColumnTransformer/OneHotEncoder).

        - Nur DataFrame: andere Inputs werden unverändert zurückgegeben.
        - Kopiert X (kein Inplace).
        - Vereinheitlicht Missing Values: pd.NaT / pd.NA -> np.nan.
        - Casts problematische pandas-Dtypes:
          * StringDtype ('string[...]') -> object
          * category -> object
          * nullable boolean ('boolean') -> float64 (True/False -> 1.0/0.0, NA -> nan)
          * nullable integers ('Int*') -> float64 (NA -> nan)

        Rückgabe: bereinigtes DataFrame für stabile sklearn-Pipelines.
        """

    if not isinstance(X, pd.DataFrame):
        return X

    X = X.copy()

    # Normalize missing markers
    X = X.replace({pd.NaT: np.nan})

    for c in X.columns:
        s = X[c]
        dt = s.dtype

        # 1) FORCE pandas StringDtype (including arrow-backed) -> object
        # Covers 'string', 'string[python]', 'string[pyarrow]'
        if str(dt).startswith("string"):
            X[c] = s.astype("object")

        # 2) pandas category -> object (for OneHotEncoder pipeline)
        elif isinstance(dt, pd.CategoricalDtype):
            X[c] = s.astype("object")

        # 3) object columns: ensure pd.NA -> np.nan
        if X[c].dtype == "object":
            X[c] = X[c].where(pd.notna(X[c]), np.nan)

        # 4) nullable boolean -> float
        elif str(dt) == "boolean":
            X[c] = s.astype("float64")

        # 5) nullable integer -> float
        elif str(dt).startswith("Int"):
            X[c] = s.astype("float64")

    return X
