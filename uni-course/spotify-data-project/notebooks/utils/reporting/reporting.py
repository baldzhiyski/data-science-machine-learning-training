from __future__ import annotations
import json
from pathlib import Path
from joblib import dump
from typing import Any, Dict

"""
Reporting & Artefakt-Speicherung.

Aufgabe:
- Speichert Modelle (joblib)
- Speichert Reports/Configs (json)
- Erstellt Ordnerstrukturen

Wichtig:
Kein Training, keine Datenaufbereitung.
Nur persistieren.
"""

def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_joblib(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(obj, path)


def save_best_params(best: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
