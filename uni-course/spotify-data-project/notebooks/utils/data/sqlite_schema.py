from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import sqlite3


def list_user_tables(con: sqlite3.Connection) -> list[str]:
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
    """
    return pd.read_sql(q, con)["name"].tolist()


def get_table_info(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql(f"PRAGMA table_info({table});", con)


def get_rowcount(con: sqlite3.Connection, table: str):
    try:
        return pd.read_sql(f"SELECT COUNT(*) AS n FROM {table};", con).iloc[0, 0]
    except Exception:
        return None


def build_db_summary(con: sqlite3.Connection, max_preview_cols: int = 8):
    tables = list_user_tables(con)
    rows = []
    column_details: Dict[str, pd.DataFrame] = {}

    for t in tables:
        info = get_table_info(con, t)
        column_details[t] = info

        # Build preview: first N columns
        preview_parts = []
        for _, r in info.head(max_preview_cols).iterrows():
            preview_parts.append(
                f"{r['name']} ({r['type']})" + (" [PK]" if r["pk"] == 1 else "")
            )

        more = ""
        if len(info) > max_preview_cols:
            more = f" … (+{len(info) - max_preview_cols} more)"

        rowcount = get_rowcount(con, t)

        rows.append({
            "table": t,
            "rowcount": rowcount,
            "n_columns": int(len(info)),
            "columns_preview": ", ".join(preview_parts) + more,
        })

    summary = pd.DataFrame(rows).sort_values("rowcount", ascending=False)
    return summary, column_details