from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


def onehot_encoder_compat():
    """
    Factory für OneHotEncoder.
    - handle_unknown="ignore": verhindert Fehler bei Kategorien, die im Train nicht gesehen wurden.
    - Absichtlich minimal gehalten, um über sklearn-Versionen stabil zu bleiben.
    """
    return OneHotEncoder(handle_unknown="ignore")


def build_preprocessor_tree(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Preprocessing für Tree-Modelle (z. B. XGBoost/RandomForest):

    - Numeric: SimpleImputer(median)
    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder
    - Kein Scaling (Tree-Modelle brauchen i. d. R. kein Feature-Scaling)

    Rückgabe:
    - pre: ColumnTransformer
    - numeric_cols: Liste numerischer Spalten
    - categorical_cols: Liste kategorialer Spalten
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder_compat()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre, numeric_cols, categorical_cols


def build_preprocessor_linear(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Preprocessing für lineare Modelle (z. B. SGDRegressor/SGDClassifier/Ridge):

    - Numeric: SimpleImputer(median) + StandardScaler(with_mean=False)
      (with_mean=False ist wichtig, wenn die Matrix sparse wird)
    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder
    - Scaling ist bei linearen Modellen typischerweise hilfreich.

    Rückgabe:
    - pre: ColumnTransformer
    - numeric_cols: Liste numerischer Spalten
    - categorical_cols: Liste kategorialer Spalten
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # wichtig wegen Sparse-Matrix
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder_compat()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre, numeric_cols, categorical_cols
