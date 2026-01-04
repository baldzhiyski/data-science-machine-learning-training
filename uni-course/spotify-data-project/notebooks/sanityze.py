import numpy as np
import pandas as pd


def sklearn_sanitize_df(X):
    import numpy as np
    import pandas as pd

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
