# notebooks/utils/eda_plots.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class NumericProfileParams:
    """
    Parameter für die numerischen EDA-Plots.
    """
    bins: int = 40
    clip_q: tuple[float, float] = (0.01, 0.99)
    log_mode: str | bool = "auto"   # "auto", True, False
    show_box: bool = True
    dpi: int = 200
    chunk_size: int = 6


def as_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Konvertiert eine Spalte robust zu numeric (coerce) und ersetzt inf/-inf mit NaN.
    """
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def should_use_log(s: pd.Series, log_mode: str | bool = "auto") -> bool:
    """
    Heuristik: log1p-Ansicht nutzen, wenn heavy tail vorliegt und Werte nicht-negativ sind.
    """
    if log_mode is True:
        return True
    if log_mode is False:
        return False

    s = s.dropna()
    if s.empty:
        return False
    if s.min() < 0:
        return False

    p50 = s.quantile(0.50)
    p99 = s.quantile(0.99)
    if p50 <= 0:
        return False
    return (p99 / p50) >= 10


def chunked(items: Sequence[str], size: int) -> list[list[str]]:
    """
    Teilt eine Liste/Sequence in gleich große Chunks auf.
    """
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def numeric_profile_grid(
    df: pd.DataFrame,
    cols: list[str],
    title: str,
    *,
    params: Optional[NumericProfileParams] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Pro Feature:
      - links: Histogram (geclippt auf Quantile)
      - rechts: log1p-Hist (bei heavy tails) ODER Boxplot

    Hinweis: Wir casten explizit auf float64 vor clip(), um pandas FutureWarning bzgl.
    Downcasting zu vermeiden.
    """
    p = params or NumericProfileParams()

    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("Keine gültigen Spalten gefunden.")
        return

    nrows = len(cols)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.2 * nrows), squeeze=False)

    for i, col in enumerate(cols):
        s = as_numeric_series(df, col)
        s_nonnull = s.dropna()

        missing_rate = 1.0 - (len(s_nonnull) / max(len(s), 1))
        zero_rate = float((s_nonnull == 0).mean()) if len(s_nonnull) else np.nan

        axL, axR = axes[i, 0], axes[i, 1]

        if s_nonnull.empty:
            axL.set_title(f"{col} (all missing)")
            axL.axis("off")
            axR.axis("off")
            continue

        q_low = s_nonnull.quantile(p.clip_q[0])
        q_high = s_nonnull.quantile(p.clip_q[1])

        # ---- FIX: stabiler dtype (verhindert FutureWarning beim clip/downcasting)
        s_nonnull = pd.to_numeric(s_nonnull, errors="coerce").astype("float64")
        s_clip = s_nonnull.clip(q_low, q_high)

        use_log = should_use_log(s_nonnull, log_mode=p.log_mode)

        # LEFT: clipped histogram
        axL.hist(s_clip.values, bins=p.bins)
        axL.set_ylabel("count")
        axL.set_title(
            f"{col} | miss={missing_rate:.1%} | zero={zero_rate:.1%} | clip=[{q_low:.3g},{q_high:.3g}]"
        )

        # RIGHT: log1p hist or boxplot
        if use_log:
            s_log = np.log1p(s_nonnull.clip(lower=0))
            axR.hist(s_log.values, bins=p.bins)
            axR.set_title(f"{col} (log1p view)")
        else:
            if p.show_box:
                axR.boxplot(s_nonnull.values, vert=False, showfliers=True)
                axR.set_title(f"{col} (boxplot)")
            else:
                axR.axis("off")

        for ax in (axL, axR):
            ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=p.dpi, bbox_inches="tight")
        plt.close(fig)
        print("saved:", save_path.name)
    else:
        plt.show()


def corr_with_popularity(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if "popularity" not in df.columns:
        return pd.DataFrame()

    num = df.select_dtypes(include=["number"]).copy()
    if "popularity" not in num.columns:
        return pd.DataFrame()

    y = pd.to_numeric(num["popularity"], errors="coerce")

    rows = []
    for col in num.columns:
        if col == "popularity":
            continue  # remove self-corr

        x = pd.to_numeric(num[col], errors="coerce")

        # pairwise valid rows
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < 30:  # too small -> skip (tune if needed)
            continue

        x2 = x[m]
        y2 = y[m]

        # skip constant features
        if x2.nunique(dropna=True) <= 1:
            continue

        pear_r, pear_p = pearsonr(x2, y2)
        spear_r, spear_p = spearmanr(x2, y2)

        rows.append({
            "table": table_name,
            "feature": col,
            "n_pairs": n,
            "pearson_r": pear_r,
            "pearson_p": pear_p,
            "spearman_r": spear_r,
            "spearman_p": spear_p,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # sort by absolute correlation (strongest relationship)
    out["abs_pearson"] = out["pearson_r"].abs()
    out["abs_spearman"] = out["spearman_r"].abs()
    return out.sort_values(["abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)

def small_scatter(df, x, y, title, out_dir: Path, filename: str = None):
    # skip if columns missing
    if x not in df.columns or y not in df.columns:
        print(f"skip (missing col): {title} [{x}, {y}]")
        return

    fig = plt.figure(figsize=(5, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
    plt.title(title)
    plt.tight_layout()

    # safe filename
    if filename is None:
        safe = title.lower().replace(" ", "_").replace("/", "_").replace(".", "")
        filename = f"{safe}.png"

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path)


def top_corr_pairs(df: pd.DataFrame, k: int = 12, method: str = "spearman"):
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] < 2:
        return []

    corr = num.corr(method=method)
    # take upper triangle without diagonal
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "x", "level_1": "y", 0: "corr"})
    )
    pairs["abs_corr"] = pairs["corr"].abs()
    return pairs.sort_values("abs_corr", ascending=False).head(k)[["x", "y", "corr"]].values.tolist()


def run_profile_reports(
    *,
    out_dir: Path,
    sample_name: str,
    reports: list[tuple[str, pd.DataFrame, list[str], str]],
    params: Optional[NumericProfileParams] = None,
) -> None:
    """
        Führt mehrere Profil-Reports aus, chunked nach params.chunk_size, und speichert PNGs.

        Rückgabe:
          Liste der gespeicherten PNG-Pfade (damit du im Notebook ggf. eine kleine Übersicht bauen kannst).
        """
    p = params or NumericProfileParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for base_name, df, cols, title in reports:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        for j, cols_chunk in enumerate(chunked(cols, p.chunk_size), start=1):
            fname = f"{base_name}_profiles_p{j}.png"
            save_path = out_dir / fname
            numeric_profile_grid(
                df,
                cols_chunk,
                f"{title} (Teil {j})",
                params=p,
                save_path=save_path,
            )
