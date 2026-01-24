from typing import List

import pandas as pd


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
