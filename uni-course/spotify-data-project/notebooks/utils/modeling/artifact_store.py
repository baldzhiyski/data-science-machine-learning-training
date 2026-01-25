# utils/modeling/artifact_store.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ArtifactStoreConfig:
    out_dir: Path
    parquet_compression: str = "snappy"
    write_manifest: bool = True
    overwrite: bool = True  # if False -> skip existing files


class ArtifactStore:
    """
    Speichert ML-Artefakte konsistent:
    - DataFrames (X, datasets, panels, multi-hot)
    - Series (y)
    - schreibt optional ein Manifest (CSV/JSON-ähnlich als parquet-free Tabelle)
    """

    def __init__(self, cfg: ArtifactStoreConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_rows: list[dict] = []

    # ---------------------------
    # Public API
    # ---------------------------

    def save_df(self, name: str, df: pd.DataFrame, *, index: bool = False) -> Path:
        return self._save_table(name=name, obj=df, index=index)

    def save_series(self, name: str, s: pd.Series, *, index: bool = False, col_name: Optional[str] = None) -> Path:
        # store series as 1-col df (stable parquet schema)
        col = col_name or name
        df = pd.DataFrame({col: s})
        return self._save_table(name=name, obj=df, index=index)

    def save_optional_df(self, name: str, obj: Any, *, index: bool = False) -> Optional[Path]:
        if isinstance(obj, pd.DataFrame):
            return self.save_df(name, obj, index=index)
        return None

    def save_optional_series(self, name: str, obj: Any, *, index: bool = False, col_name: Optional[str] = None) -> Optional[Path]:
        if isinstance(obj, pd.Series):
            return self.save_series(name, obj, index=index, col_name=col_name)
        return None

    def finalize(self) -> Optional[Path]:
        if not self.cfg.write_manifest or not self._manifest_rows:
            return None
        manifest = pd.DataFrame(self._manifest_rows)
        p = self.cfg.out_dir / "_manifest.parquet"
        manifest.to_parquet(p, index=False)
        return p

    # ---------------------------
    # Internals
    # ---------------------------

    def _save_table(self, name: str, obj: pd.DataFrame, *, index: bool) -> Path:
        fname = f"{name}.parquet"
        path = self.cfg.out_dir / fname

        if path.exists() and not self.cfg.overwrite:
            self._add_manifest(name, path, obj, status="skipped_exists")
            return path

        obj.to_parquet(path, index=index, compression=self.cfg.parquet_compression)
        self._add_manifest(name, path, obj, status="written")
        return path

    def _add_manifest(self, name: str, path: Path, obj: pd.DataFrame, *, status: str) -> None:
        self._manifest_rows.append({
            "name": name,
            "file": path.name,
            "path": str(path.resolve()),
            "status": status,
            "n_rows": int(obj.shape[0]),
            "n_cols": int(obj.shape[1]),
            "cols_preview": ",".join(list(obj.columns[:12])),
        })