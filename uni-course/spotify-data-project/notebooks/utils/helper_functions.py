import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import json
import ast
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)


def safe_get(name: str):
    return globals().get(name, None)


#%%
def parse_datetime_from_candidates(df: pd.DataFrame, candidates: list[str], out_col: str) -> pd.DataFrame:
    """
    Pick the first existing column from `candidates` and parse it to datetime.
    If the chosen column is already datetime-like, keep it as-is.
    If none exists, create out_col as NaT.
    """
    src = None
    for c in candidates:
        if c in df.columns:
            src = c
            break

    if src is None:
        df[out_col] = pd.NaT
        return df

    s = df[src]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[out_col] = s
    else:
        df[out_col] = pd.to_datetime(s.astype("string"), errors="coerce")

    return df

def col_or_na(df: pd.DataFrame, col: str, dtype: Optional[str] = None) -> pd.Series:
    """
    Zweck:
    - Sichere Hilfsfunktion, um eine Spalte aus einem DataFrame zu holen.
    - Falls die Spalte nicht existiert, wird stattdessen eine Series mit NA-Werten (gleiche Länge/Index) zurückgegeben.
    Warum?
    - In realen Datensätzen fehlen manchmal Spalten (z.B. bei Batch-Scoring).
    - So verhindert man KeyErrors und die Pipeline bleibt robust.

    Parameter:
    - df: pandas DataFrame
    - col: Name der gewünschten Spalte
    - dtype: optionaler Ziel-Datentyp (z.B. "float64", "string")

    Rückgabe:
    - Immer eine pandas Series (nie None).
    """
    # Sicherheitscheck: df muss wirklich ein DataFrame sein, sonst ist die Nutzung fehlerhaft.
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("col_or_na: df must be a pandas DataFrame")

    # Falls die Spalte existiert: hole sie.
    if col in df.columns:
        s = df[col]

        # Optional: versuche die Series auf den gewünschten dtype zu casten.
        # Wenn das Casting scheitert, ignorieren wir den Fehler (Pipeline soll nicht abbrechen).
        if dtype is not None:
            try:
                s = s.astype(dtype)
            except Exception:
                pass

        return s

    # Falls die Spalte fehlt: gib eine NA-Serie zurück, aber mit gleichem Index wie df.
    # Vorteil: Downstream-Code kann weiterhin damit rechnen, dass die Länge passt.
    return pd.Series(pd.NA, index=df.index)


def safe_len_series(s: pd.Series) -> pd.Series:
    """
    Zweck:
    - Berechnet die Länge (Anzahl Zeichen) jedes Eintrags in einer Textspalte.
    - Null/NaN werden zu "" gemacht, damit keine Fehler entstehen.

    Ablauf:
    1) In String-Typ umwandeln (damit .str len sicher funktioniert)
    2) NaNs mit "" ersetzen
    3) Zeichenlänge berechnen
    4) Als int32 zurückgeben (spart Speicher bei großen Datenmengen)
    """
    return s.astype("string").fillna("").str.len().astype("int32")


def safe_word_count_series(s: pd.Series) -> pd.Series:
    """
    Zweck:
    - Zählt die Anzahl Wörter pro Eintrag in einer Textspalte.
    - Null/NaN werden zu "" gemacht, damit keine Fehler entstehen.

    Ablauf:
    1) In String-Typ umwandeln
    2) NaNs mit "" ersetzen
    3) Text splitten (Standard: whitespace)
    4) Anzahl Tokens/Listelemente zählen
    5) Als int32 zurückgeben
    """
    return s.astype("string").fillna("").str.split().str.len().astype("int32")


def add_release_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Zweck:
    - Erzeugt aus einer Datums-Spalte zusätzliche Zeit-Features:
      * release_year   (Jahr)
      * release_month  (Monat)
      * release_decade (Dekade, z.B. 1990, 2000, 2010)

    Robustheit:
    - Unterstützt echte Datumsstrings (YYYY / YYYY-MM / YYYY-MM-DD)
    - Unterstützt Unix-Epoch als Zahl (Sekunden oder Millisekunden), auch NEGATIV (vor 1970),
      z.B. -473385600000

    Parameter:
    - df: DataFrame
    - date_col: Name der Spalte mit Datum/Datetime-ähnlichen Werten

    Rückgabe:
    - Kopie von df, erweitert um neue Spalten (Original bleibt unverändert).
    """
    df = df.copy()

    s = col_or_na(df, date_col)

    # --- 1) Versuche numeric epoch parsing (ms oder s) ---
    num = pd.to_numeric(s, errors="coerce")
    dt_num = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    is_num = num.notna()
    # Heuristik: ms ~ 1e12, s ~ 1e9 (heute); alles >= 1e10 behandeln wir als ms
    is_ms = is_num & (num.abs() >= 1e10)
    is_s  = is_num & ~is_ms

    if is_ms.any():
        dt_num.loc[is_ms] = pd.to_datetime(num.loc[is_ms], unit="ms", origin="unix", errors="coerce")
    if is_s.any():
        dt_num.loc[is_s] = pd.to_datetime(num.loc[is_s], unit="s", origin="unix", errors="coerce")

    # --- 2) Fallback: String parsing (ISO-like) ---
    dt_str = pd.to_datetime(s.astype("string"), errors="coerce")

    # Combine: prefer numeric epoch parse, fallback to string parse
    dt = dt_num.combine_first(dt_str)

    # Features (nullable Int64)
    df["release_year"] = dt.dt.year.astype("Int64")
    df["release_month"] = dt.dt.month.astype("Int64")
    df["release_decade"] = ((dt.dt.year // 10) * 10).astype("Int64")

    return df


def log1p_numeric(s: pd.Series) -> pd.Series:
    """
    Zweck:
    - Wendet log(1 + x) auf numerische Werte an.
    Warum?
    - Viele numerische Features sind stark schief verteilt (z.B. Streams, Follower, Counts).
    - log1p reduziert Ausreißer-Einfluss und macht Verteilungen "normaler" -> oft besser für ML.

    Details:
    - pd.to_numeric(..., errors="coerce") macht aus nicht-numerischen Werten NaN.
    - np.log1p(x) ist stabil für x=0 (log(1)=0).
    """
    x = pd.to_numeric(s, errors="coerce")
    return np.log1p(x).astype("float64")


def ensure_list_column(s: pd.Series) -> pd.Series:
    """
    Zweck:
    - Stellt sicher, dass jede Zelle in der Spalte eine echte Python-Liste ist.
    - Sehr wichtig für Features wie Genres/Tags, die manchmal als Strings gespeichert sind.

    Akzeptierte Eingaben pro Zelle:
    - bereits echte Listen: ["pop", "rock"]
    - JSON-String: '["pop","rock"]'
    - Python-repr-String: "['pop', 'rock']"
    - None/NaN: wird zu []

    Rückgabe:
    - Series, in der jede Zeile garantiert eine Liste (oder []) enthält.
    """

    def parse_one(v):
        # Fall 1: schon eine Liste -> direkt zurück.
        if isinstance(v, list):
            return v

        # Fall 2: None oder NaN -> leere Liste.
        # np.isnan funktioniert nur für floats; daher prüfen wir vorher den Typ.
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []

        # Fall 3: String -> mehrere Parser-Versuche.
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []

            # Versuch A: JSON parsen (sicherer als eval).
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            # Versuch B: Python-Literal parsen (z.B. "['a','b']").
            # ast.literal_eval ist viel sicherer als eval (führt keinen Code aus).
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        # Wenn alles fehlschlägt: leere Liste als Default (robust für Pipelines).
        return []

    return s.apply(parse_one)


def top_k_list_counts(list_series: pd.Series, top_k: int) -> List[str]:
    """
    Zweck:
    - Ermittelt die Top-K häufigsten Elemente über alle Listen hinweg.
    Beispiel:
    - Zeile1: ["pop", "rock"]
    - Zeile2: ["pop"]
    -> pop:2, rock:1  => top_k=1 => ["pop"]

    Warum?
    - Für Multi-Hot-Encoding braucht man oft eine feste Auswahl von Kategorien
      (z.B. die häufigsten Genres), damit Feature-Spalten stabil bleiben.

    Rückgabe:
    - Liste der Top-K Labels (als Strings) in Häufigkeits-Reihenfolge.
    """
    from collections import Counter

    c = Counter()

    # Iteriere über jede Zeile (jede "Liste" pro Song/Item).
    for lst in list_series:
        if isinstance(lst, list):
            for x in lst:
                # pd.notna stellt sicher, dass wir keine NA/NaN zählen.
                if pd.notna(x):
                    c[str(x)] += 1

    # most_common liefert (label, count)-Paare; wir geben nur die Labels zurück.
    return [k for k, _ in c.most_common(top_k)]


def genres_to_multihot(df: pd.DataFrame, list_col: str, top_genres: List[str], prefix: str) -> pd.DataFrame:
    """
    Zweck:
    - Wandelt eine Listen-Spalte (z.B. Genres pro Song) in Multi-Hot-Features um.
      Multi-Hot bedeutet: pro Genre eine Spalte, Werte 0/1.
    Beispiel top_genres=["pop","rock"]:
      ["pop"]      -> pop=1 rock=0
      ["pop","rock"] -> pop=1 rock=1

    Parameter:
    - df: DataFrame
    - list_col: Spaltenname, der Listen enthält (z.B. "genres_list")
    - top_genres: feste Liste der erlaubten Genres (Encoding-Vertrag!)
    - prefix: Prefix für die Spaltennamen (z.B. "spotify_")

    Rückgabe:
    - DataFrame nur mit Multi-Hot-Spalten, gleicher Index wie df.
    """
    # Wenn keine Genres definiert sind: gib leeren DF zurück (aber mit passendem Index).
    if not top_genres:
        return pd.DataFrame(index=df.index)

    # Matrix mit 0 initialisieren: Zeilen = Datensätze, Spalten = Genres.
    # int8 spart Speicher (0/1 reicht).
    m = np.zeros((len(df), len(top_genres)), dtype=np.int8)

    # Mapping Genre -> Spaltenindex für schnelles Nachschlagen.
    idx = {g: i for i, g in enumerate(top_genres)}

    # Hole die Listen aus der DataFrame-Spalte.
    lists = df[list_col]

    # Für jede Zeile r: setze die passenden Genre-Spalten auf 1.
    for r, lst in enumerate(lists):
        if isinstance(lst, list):
            for g in lst:
                j = idx.get(str(g))
                if j is not None:
                    m[r, j] = 1

    # Baue einen DataFrame mit sprechenden Spaltennamen.
    return pd.DataFrame(m, columns=[f"{prefix}genre_{g}" for g in top_genres])


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


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Zweck:
    - Berechnet Standard-Metriken für Regressionsmodelle.

    Metriken:
    - MAE  (Mean Absolute Error): durchschnittlicher absoluter Fehler
    - RMSE (Root Mean Squared Error): bestraft große Fehler stärker (Quadrat)
    - R2   (Bestimmtheitsmaß): erklärt wie viel Varianz durch das Modell erklärt wird

    Rückgabe:
    - Dictionary mit Metriken als float (praktisch für Logging/JSON).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_report_binary(y_true, y_proba, threshold=0.5) -> Dict[str, Any]:
    """
    Zweck:
    - Report für binäre Klassifikation, wenn das Modell Wahrscheinlichkeiten liefert.

    Parameter:
    - y_true: echte Labels (0/1)
    - y_proba: vorhergesagte Wahrscheinlichkeit für Klasse 1
    - threshold: Schwelle, ab wann 1 vorhergesagt wird (default 0.5)

    Output:
    - roc_auc: ROC-AUC (nur sinnvoll, wenn beide Klassen vorkommen)
    - pr_auc:  PR-AUC (Average Precision), oft wichtig bei Imbalance
    - f1:      F1-Score basierend auf threshold
    - confusion_matrix: [[TN, FP],[FN, TP]] als Liste (gut für JSON)

    Extra:
    - Wenn y_true nur eine Klasse enthält, sind AUC/F1 nicht sinnvoll -> None.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Aus Wahrscheinlichkeit wird harte Vorhersage 0/1 basierend auf Schwelle.
    y_pred = (y_proba >= threshold).astype(int)

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return out


def best_f1_threshold(y_true, proba, thresholds=np.linspace(0.05, 0.95, 19)):
    """
    Zweck:
    - Findet die Threshold-Schwelle, die den besten F1-Score liefert.

    Warum?
    - 0.5 ist nicht immer optimal (besonders bei unbalancierten Klassen).
    - Man scannt mehrere Thresholds und nimmt den besten nach F1.

    Parameter:
    - y_true: echte Labels (0/1)
    - proba: Modellwahrscheinlichkeiten für Klasse 1
    - thresholds: Liste/Array an Schwellenwerten (Default: 0.05..0.95)

    Rückgabe:
    - best_t: bester Threshold
    - best_f1: bester F1-Score, der damit erreicht wird
    """
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)

        # Wenn der neue F1 besser ist, merken wir uns die Schwelle.
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return float(best_t), float(best_f1)

def ensure_list_fast(x):
    """Fast list conversion."""
    if isinstance(x, list): return x
    if isinstance(x, str):
        s = x.strip()
        if not s: return []
        if "|" in s: return [p.strip() for p in s.split("|") if p.strip()]
        if "," in s: return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return []


def pick_release_cols_fast(df: pd.DataFrame) -> Tuple[str, str]:
    if {"release_year", "release_month"}.issubset(df.columns):
        return "release_year", "release_month"
    raise ValueError("Need release_year + release_month columns.")



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
