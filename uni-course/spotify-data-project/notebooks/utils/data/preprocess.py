from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ModelKind = Literal["tree", "linear"]


@dataclass
class TabularPreprocessor:
    """
    Reusable preprocessor builder for mixed tabular data.

    - model_kind="tree": numeric impute only, categorical one-hot
    - model_kind="linear": numeric impute + scaling, categorical one-hot
    - optional text_cols: TF-IDF per text column

    Usage:
        pre = TabularPreprocessor(model_kind="tree", text_cols=["title", "lyrics"])
        ct = pre.build(X_train)
        Xtr = ct.fit_transform(X_train)
        Xva = ct.transform(X_val)
    """
    model_kind: ModelKind = "tree"
    text_cols: Optional[List[str]] = None

    # TF-IDF knobs (safe defaults)
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2

    # ColumnTransformer config
    sparse_threshold: float = 0.3
    remainder: str = "drop"

    def onehot_encoder(self) -> OneHotEncoder:
        return OneHotEncoder(handle_unknown="ignore")

    def infer_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        text_cols = [c for c in (self.text_cols or []) if c in X.columns]

        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if c not in numeric_cols and c not in text_cols]

        return numeric_cols, categorical_cols, text_cols

    def _numeric_pipe(self) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.model_kind == "linear":
            # with_mean=False is important for sparse matrices
            steps.append(("scaler", StandardScaler(with_mean=False)))
        return Pipeline(steps=steps)

    def _categorical_pipe(self) -> Pipeline:
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", self.onehot_encoder()),
        ])

    def _text_pipe(self) -> Pipeline:
        # Vectorizer expects 1D strings; we impute missing values to ""
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("tfidf", TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                min_df=self.tfidf_min_df,
            )),
        ])

    def build(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols, categorical_cols, text_cols = self.infer_columns(X)

        transformers = []
        if numeric_cols:
            transformers.append(("num", self._numeric_pipe(), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", self._categorical_pipe(), categorical_cols))

        # TF-IDF per text column (important)
        for c in text_cols:
            transformers.append((f"txt_{c}", self._text_pipe(), c))

        return ColumnTransformer(
            transformers=transformers,
            remainder=self.remainder,
            sparse_threshold=self.sparse_threshold,
        )