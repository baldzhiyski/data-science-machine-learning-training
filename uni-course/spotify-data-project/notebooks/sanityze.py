import numpy as np
import pandas as pd


def sklearn_sanitize_df(X):
    if not isinstance(X, pd.DataFrame):
        return X

    X = X.copy()

    # Convert NaT -> np.nan
    X = X.replace({pd.NaT: np.nan})

    for c in X.columns:
        dt = X[c].dtype

        # pandas string or categorical -> object + np.nan
        if pd.api.types.is_string_dtype(dt) or isinstance(dt, pd.CategoricalDtype):
            X[c] = X[c].astype("object")
            X[c] = X[c].where(pd.notna(X[c]), np.nan)

        # pandas nullable boolean -> float (0/1/nan)
        elif str(dt) == "boolean":
            X[c] = X[c].astype("float64")

        # pandas nullable integer (Int64, Int32...) -> float (so missing -> np.nan)
        elif str(dt).startswith("Int"):
            X[c] = X[c].astype("float64")

        # object columns might still contain pd.NA -> replace with np.nan
        elif X[c].dtype == "object":
            X[c] = X[c].where(pd.notna(X[c]), np.nan)

    return X
