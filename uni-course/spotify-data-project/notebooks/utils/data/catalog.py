# utils/datasets_modeling/catalog.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class TableCatalogConfig:
    """Konfiguration für den Zugriff auf die Clean-Layer."""
    base_dir: Path
    tables: List[str]


class TableCatalog:
    """
    Zugriffsschicht für Clean-Layer Tabellen (Parquet).
    Hält die Daten in einem Dict und bietet Helper für dtypes/IDs.
    """

    def __init__(self, config: TableCatalogConfig):
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}

    def load(self, strict: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Lädt Tabellen aus Parquet.
        - strict=True: wirft Fehler, wenn eine Tabelle fehlt
        - strict=False: lädt nur vorhandene Tabellen
        """
        missing = []
        out: Dict[str, pd.DataFrame] = {}

        for name in self.config.tables:
            fp = self.config.base_dir / f"{name}.parquet"
            if fp.exists():
                out[name] = pd.read_parquet(fp)
            else:
                missing.append(name)

        if strict and missing:
            raise FileNotFoundError(f"Fehlende Tabellen in Clean-Layer: {missing}")

        self.data = out
        return out

    @staticmethod
    def ensure_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Sorgt dafür, dass ID/FK-Spalten überall string sind (join-sicher)."""
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype("string")
        return df

    def get(self, name: str) -> pd.DataFrame:
        """Sichere Abfrage aus dem Katalog."""
        if name not in self.data:
            raise KeyError(f"Tabelle '{name}' nicht geladen.")
        return self.data[name]