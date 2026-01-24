# utils/paths.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json, time, platform, os

import numpy as np
import pandas as pd


_DEFAULT_ROOT = r"C:\GitHub\uni-project-metrics-and-data"
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

# Optional convenience (raw dir/db)
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_SPOTIFY_DB_PATH = DATA_RAW_DIR / "spotify.sqlite"


def load_sample_name(
    base_export_dir: Path = PROJECT_ROOT / "data" / "interim" / "converted_sqlite_samples",
    current_sample_file: str = "current_sample.json",
) -> str:
    """
    Reads the current active sample name from interim/converted_sqlite_samples/current_sample.json.
    Keeps compatibility with your existing workflow.
    """
    cfg_path = base_export_dir / current_sample_file
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg["SAMPLE_NAME"]


@dataclass(frozen=True)
class Paths:
    """
    Central path registry for one experiment run / one sample.

    IMPORTANT:
    - Keep these field names stable (backwards compatibility).
    - New fields are appended (won't break old code that uses attributes).
    """
    sample_name: str

    clean_parquet_dir: Path
    clean_csv_dir: Path
    modeling_dir: Path
    models_dir: Path
    reports_dir: Path
    input_targets_path: Path
    raw_dir: Path
    reports_dir_cleaning: Path
    tuned_models_dir: Path
    reports_dir_tuned: Path
    interim_samples_dir: Path
    export_dir: Path
    schema_reports_dir: Path
    meta_path: Path
    current_sample_pointer: Path

    # Convenience (raw DB)
    raw_spotify_db_path: Path


def make_paths(sample_name: str) -> Paths:
    """
    Builds all project paths for a specific sample name.

    This is a drop-in replacement that preserves your old directories
    and adds new ones needed for Notebook 1 (sampling/export).
    """

    # ---- Existing dirs (unchanged) ----
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

    interim_samples_dir = PROJECT_ROOT / "data" / "interim" / "converted_sqlite_samples"
    export_dir = raw_dir
    schema_reports_dir = PROJECT_ROOT / "data" / "reports" / "schema_overview" / sample_name
    meta_path = interim_samples_dir / "current_sample.json"
    current_sample_pointer = interim_samples_dir

    return Paths(
        sample_name=sample_name,


        input_targets_path=input_targets_path,
        clean_parquet_dir=clean_parquet_dir,
        clean_csv_dir=clean_csv_dir,
        raw_dir=raw_dir,
        modeling_dir=modeling_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        reports_dir_cleaning=reports_dir_cleaning,
        tuned_models_dir=tuned_models_dir,
        reports_dir_tuned=reports_dir_tuned,
        interim_samples_dir=interim_samples_dir,
        export_dir=export_dir,
        schema_reports_dir=schema_reports_dir,
        meta_path=meta_path,
        current_sample_pointer=current_sample_pointer,
        raw_spotify_db_path=RAW_SPOTIFY_DB_PATH,
    )


def ensure_dirs(paths: Paths) -> None:
    """
    Creates all directories needed for the run (safe to call repeatedly).
    """
    to_create = [
        # existing
        paths.modeling_dir,
        paths.reports_dir_cleaning,
        paths.models_dir,
        paths.reports_dir,
        paths.input_targets_path,
        paths.raw_dir,
        paths.clean_parquet_dir,
        paths.clean_csv_dir,
        paths.tuned_models_dir,
        paths.reports_dir_tuned,

        # new
        paths.interim_samples_dir,
        paths.schema_reports_dir,
        paths.export_dir,
    ]

    for p in to_create:
        p.mkdir(parents=True, exist_ok=True)


def build_run_meta(
    paths: Paths,
    *,
    random_seed: int,
    allow_leaky_features: bool,
    main_album_strategy: str,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Creates a compact run metadata dict for saving as JSON.

    Adds:
    - versions (python, numpy, pandas)
    - key run flags
    - all relevant paths (as strings)
    """
    meta = {
        "run_ts_unix": int(time.time()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_seed": random_seed,
        "allow_leaky_features": allow_leaky_features,
        "main_album_strategy": main_album_strategy,

        # include all paths
        "paths": {k: str(v) for k, v in asdict(paths).items() if k != "sample_name"},
        "sample_name": paths.sample_name,
    }

    if extra:
        meta["extra"] = extra

    return meta