import pandas as pd
import numpy as np


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



