# utils/paths.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json, time, platform

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Absolute project root (Windows)
# ------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\GitHub\uni-project-metrics-and-data").resolve()

# Optional convenience (raw dir/db)
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_SPOTIFY_DB_PATH = DATA_RAW_DIR / "spotify.sqlite"


def load_sample_name(
    base_export_dir: Path = PROJECT_ROOT / "data" / "interim" / "converted_sqlite_samples",
    current_sample_file: str = "current_sample.json",
) -> str:
    cfg_path = base_export_dir / current_sample_file
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg["SAMPLE_NAME"]


@dataclass(frozen=True)
class Paths:
    sample_name: str

    clean_parquet_dir: Path
    clean_csv_dir: Path
    modeling_dir: Path
    models_dir: Path
    reports_dir: Path
    input_targets_path: Path
    raw_dir: Path
    reports_dir_cleaning:Path
    tuned_models_dir: Path
    reports_dir_tuned: Path


def make_paths(sample_name: str) -> Paths:
    # All base dirs anchored to PROJECT_ROOT (no relative paths)
    clean_parquet_dir = PROJECT_ROOT / "data" / "processed" / "parquet" / sample_name
    clean_csv_dir = PROJECT_ROOT / "data" / "processed" / "clean_csv" / sample_name
    raw_dir = PROJECT_ROOT / "data" / "interim" / "converted_sqlite_samples" / sample_name

    modeling_dir = PROJECT_ROOT / "data" / "processed" / "modeling" / sample_name
    input_targets_path = PROJECT_ROOT / "data" / "baseline_models_datasets" / sample_name
    models_dir = PROJECT_ROOT / "data" / "models" / "baseline" / sample_name
    reports_dir = PROJECT_ROOT / "data" / "reports" / "baseline_models" / sample_name
    reports_dir_cleaning = PROJECT_ROOT / "data" / "reports" / "cleaning" / sample_name

    tuned_models_dir = PROJECT_ROOT / "data" / "models" / "tuned" / sample_name
    reports_dir_tuned = PROJECT_ROOT / "data" / "reports" / "tuned_models" / sample_name

    return Paths(
        sample_name=sample_name,
        input_targets_path=input_targets_path,
        clean_parquet_dir=clean_parquet_dir,
        raw_dir=raw_dir,
        reports_dir_cleaning=reports_dir_cleaning,
        clean_csv_dir=clean_csv_dir,
        modeling_dir=modeling_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        tuned_models_dir=tuned_models_dir,
        reports_dir_tuned=reports_dir_tuned,

    )


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.modeling_dir, paths.reports_dir_cleaning,paths.models_dir, paths.reports_dir, paths.input_targets_path, paths.raw_dir,paths.clean_parquet_dir , paths.clean_csv_dir, paths.tuned_models_dir, paths.reports_dir_tuned ]:
        p.mkdir(parents=True, exist_ok=True)


def build_run_meta(
    paths: Paths,
    *,
    random_seed: int,
    allow_leaky_features: bool,
    main_album_strategy: str,
) -> dict:
    return {
        "run_ts_unix": int(time.time()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seed": random_seed,
        "allow_leaky_features": allow_leaky_features,
        "main_album_strategy": main_album_strategy,
        "paths": {k: str(v) for k, v in asdict(paths).items() if k != "sample_name"},
        "sample_name": paths.sample_name,
    }
