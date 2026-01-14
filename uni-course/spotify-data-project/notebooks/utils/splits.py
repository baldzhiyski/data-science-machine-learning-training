from __future__ import annotations
import numpy as np
import pandas as pd

"""
Splitting-Strategien (Train/Val/Test).

Enthält:
- Kohorten-basierte Zeit-Splits (z.B. cohort_ym)
- Einfache time-fraction Splits (für kleine Panels/Sequenzen)

Wichtig:
Splits müssen reproduzierbar und erklärbar sein (keine zufälligen Leaks).
"""


def cohort_time_split(meta_df: pd.DataFrame, cohort_col: str = "cohort_ym", n_val: int = 3, n_test: int = 6):
    """
        Erzeugt einen zeitlichen Kohorten-Split auf Basis eines Kohorten-Identifiers.

        Logik:
        - Sortiere Kohorten
        - letzte n_test Kohorten -> Test
        - vorherige n_val Kohorten -> Val
        - Rest -> Train

        Nutzen:
        - verhindert Leakage über Zeit
        - entspricht realistischer Vorhersage: Zukunft aus Vergangenheit
        """
    groups_sorted = np.sort(meta_df[cohort_col].dropna().unique())
    if len(groups_sorted) < (n_test + n_val + 1):
        raise ValueError(f"Not enough cohorts: {len(groups_sorted)} (need >= {n_test+n_val+1})")

    test_groups = groups_sorted[-n_test:]
    val_groups  = groups_sorted[-(n_test + n_val):-n_test]
    train_groups= groups_sorted[:-(n_test + n_val)]

    idx_tr = meta_df[cohort_col].isin(train_groups).to_numpy()
    idx_va = meta_df[cohort_col].isin(val_groups).to_numpy()
    idx_te = meta_df[cohort_col].isin(test_groups).to_numpy()
    return idx_tr, idx_va, idx_te


def time_fraction_split(n: int, val_frac: float = 0.10, test_frac: float = 0.15, min_val: int = 30, min_test: int = 30):
    """
    Einfacher zeitlicher Split basierend auf Fraktionen der Daten.
    Logik:
    - Sortiere Daten nach Zeit (implizit durch Index)
    - letzte test_frac der Daten -> Test
    - vorherige val_frac der Daten -> Val
    - Rest -> Train
    Nutzen:
    - verhindert Leakage über Zeit
    - entspricht realistischer Vorhersage: Zukunft aus Vergangenheit
    Raises:
        ValueError: Wenn nicht genug Samples für den Split vorhanden sind.
    """

    n_test = max(min_test, int(test_frac * n))
    n_val  = max(min_val, int(val_frac * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough samples for split: n={n}, train={n_train}, val={n_val}, test={n_test}")

    idx_tr = np.arange(0, n_train)
    idx_va = np.arange(n_train, n_train + n_val)
    idx_te = np.arange(n_train + n_val, n)
    return idx_tr, idx_va, idx_te
