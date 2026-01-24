import pandas as pd


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



