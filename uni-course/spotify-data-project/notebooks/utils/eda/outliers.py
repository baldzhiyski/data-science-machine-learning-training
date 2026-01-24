# utils/eda/outliers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class OutlierParams:
    q_low: float = 0.005
    q_high: float = 0.995
    use_iqr: bool = True
    iqr_k: float = 1.5
    include_indices: bool = False

    # Output options
    top_n: int = 15
    bar_top_k: int = 12
    dpi: int = 200


# ---------------------------
# Rules
# ---------------------------
def default_rules():
    """
    Domain-Rules für Spotify-ähnliche Daten.
    Jede Rule liefert eine Bool-Maske (True = invalid).
    """
    return {
        # Tracks
        "duration": lambda s: (s <= 0) | (s > 30 * 60 * 1000),
        "track_number": lambda s: (s <= 0) | (s > 200),
        "disc_number": lambda s: (s <= 0) | (s > 10),
        "popularity": lambda s: (s < 0) | (s > 100),
        "explicit": lambda s: ~s.isin([0, 1]),

        # Audio
        "tempo": lambda s: (s <= 0) | (s > 300),
        "time_signature": lambda s: ~s.isin([3, 4, 5]),
        "loudness": lambda s: (s < -40) | (s > 5),
    }


# ---------------------------
# Outlier helpers
# ---------------------------
def quantile_outliers(s: pd.Series, low_q=0.005, high_q=0.995):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series([], dtype=bool), (np.nan, np.nan)
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    mask = (s < lo) | (s > hi)
    return mask, (lo, hi)


def iqr_outliers(s: pd.Series, k=1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series([], dtype=bool), (np.nan, np.nan)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (s < lo) | (s > hi)
    return mask, (lo, hi)


def validate_release_date_epoch_ms(series, min_year=1900, max_year=None):
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if max_year is None:
        max_year = pd.Timestamp.today().year + 1

    dt = pd.to_datetime(s, unit="ms", errors="coerce")
    year = dt.dt.year
    invalid = dt.isna() | (year < min_year) | (year > max_year)
    return invalid, year


# ---------------------------
# Core report
# ---------------------------
def robust_outlier_report(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    params: Optional[OutlierParams] = None,
    rules: Optional[dict] = None,
):
    p = params or OutlierParams()
    rules = rules or default_rules()

    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()

    n_total = len(df)
    rows = []

    for col in cols:
        if col not in df.columns:
            continue

        s_full = df[col]
        if not pd.api.types.is_numeric_dtype(s_full):
            continue

        s_nonnull = pd.to_numeric(s_full, errors="coerce").dropna()
        miss_rate = 1.0 - (len(s_nonnull) / max(n_total, 1))
        zero_rate = float((s_nonnull == 0).mean()) if len(s_nonnull) else np.nan

        # Rule invalids
        invalid_count = 0
        invalid_rate = 0.0
        invalid_rule = ""

        if col == "release_date":
            inv_mask, _ = validate_release_date_epoch_ms(s_full)
            invalid_count = int(inv_mask.sum())
            invalid_rate = invalid_count / max(n_total, 1)
            invalid_rule = "epoch-ms date range"
        elif col in rules:
            inv_mask = rules[col](s_full).fillna(False)
            invalid_count = int(inv_mask.sum())
            invalid_rate = invalid_count / max(n_total, 1)
            invalid_rule = "domain rule"

        # Quantile outliers
        q_mask, (q_lo, q_hi) = quantile_outliers(s_full, p.q_low, p.q_high)
        q_count = int(q_mask.sum()) if len(q_mask) else 0
        q_rate = q_count / max(n_total, 1)

        # IQR outliers
        iqr_count = np.nan
        iqr_rate = np.nan
        iqr_lo = np.nan
        iqr_hi = np.nan
        if p.use_iqr:
            i_mask, (i_lo, i_hi) = iqr_outliers(s_full, p.iqr_k)
            iqr_count = int(i_mask.sum()) if len(i_mask) else 0
            iqr_rate = iqr_count / max(n_total, 1)
            iqr_lo, iqr_hi = i_lo, i_hi

        rows.append({
            "col": col,
            "missing_%": round(miss_rate * 100, 2),
            "zero_%": round(zero_rate * 100, 2) if pd.notna(zero_rate) else np.nan,

            "invalid_n": invalid_count,
            "invalid_%": round(invalid_rate * 100, 3),
            "invalid_rule": invalid_rule,

            "q_outliers_n": q_count,
            "q_outliers_%": round(q_rate * 100, 3),
            "q_bounds": f"[{q_lo:.3g}, {q_hi:.3g}]" if pd.notna(q_lo) else "",

            "iqr_outliers_n": iqr_count,
            "iqr_outliers_%": round(iqr_rate * 100, 3) if pd.notna(iqr_rate) else np.nan,
            "iqr_bounds": f"[{iqr_lo:.3g}, {iqr_hi:.3g}]" if pd.notna(iqr_lo) else "",
        })

    report = pd.DataFrame(rows).sort_values(
        by=["invalid_n", "q_outliers_n"],
        ascending=False
    )
    return report


# ---------------------------
# Saving + plotting helpers
# ---------------------------
def _save_top_table_png(rep: pd.DataFrame, out_path: Path, top_n: int, dpi: int):
    top = rep.head(top_n)
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(top) + 2))
    ax.axis("off")

    tbl = ax.table(
        cellText=top.values,
        colLabels=top.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_bar(rep: pd.DataFrame, col_y: str, out_path: Path, title: str, top_k: int, dpi: int):
    plot_df = rep.sort_values("invalid_n", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(plot_df["col"], plot_df[col_y])
    ax.set_title(title)
    ax.set_ylabel(col_y)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_outlier_suite(*, tables: dict[str, pd.DataFrame], out_dir: Path, params: Optional[OutlierParams] = None):
    """
    One-call entrypoint für Notebook:
    - berechnet robust_outlier_report je Tabelle
    - speichert CSV + Top-table PNG + 2 Barplots
    """
    p = params or OutlierParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = default_rules()

    for name, df in tables.items():
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            print("skip (no numeric cols):", name)
            continue

        rep = robust_outlier_report(df, cols=num_cols, params=p, rules=rules)

        csv_path = out_dir / f"{name}_robust_outlier_report.csv"
        rep.to_csv(csv_path, index=False, encoding="utf-8-sig")

        table_png = out_dir / f"{name}_robust_outlier_report_top{p.top_n}.png"
        _save_top_table_png(rep, table_png, top_n=p.top_n, dpi=p.dpi)

        bar1 = out_dir / f"{name}_invalid_percent_top{p.bar_top_k}.png"
        _save_bar(rep, "invalid_%", bar1, f"{name}: invalid_% (Top {p.bar_top_k})", p.bar_top_k, p.dpi)

        bar2 = out_dir / f"{name}_q_outliers_percent_top{p.bar_top_k}.png"
        _save_bar(rep, "q_outliers_%", bar2, f"{name}: q_outliers_% (Top {p.bar_top_k})", p.bar_top_k, p.dpi)

        print("saved:", csv_path.name, "|", table_png.name, "|", bar1.name, "|", bar2.name)