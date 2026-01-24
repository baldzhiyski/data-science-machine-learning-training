import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class DataIntegrityChecker:
    """
    Checks data integrity for a music database schema including:
    - Duplicate detection
    - Primary key uniqueness
    - Foreign key constraints
    """

    def __init__(self, data_dir: Path, schema_reports_dir: Path):
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(schema_reports_dir) / "integrity"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.tables: Dict[str, Optional[pd.DataFrame]] = {}
        self.results: List[Dict] = []

    def load_table(self, name: str) -> Optional[pd.DataFrame]:
        """Load a CSV table from the data directory."""
        path = self.data_dir / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    def load_all_tables(self) -> None:
        """Load all expected tables from the data directory."""
        table_names = [
            "tracks", "audio_features", "artists", "albums",
            "r_track_artist", "r_albums_tracks", "r_artist_genre", "genres"
        ]
        self.tables = {name: self.load_table(name) for name in table_names}

    @staticmethod
    def get_primary_key(table_name: str) -> str:
        """Get the primary key column name for a given table."""
        return "track_id" if table_name == "tracks" else "id"

    @staticmethod
    def calculate_percentage(part: int, whole: int) -> float:
        """Calculate percentage with proper handling of zero division."""
        return 0.0 if whole == 0 else round(100.0 * part / whole, 3)

    def add_result(self, table: str, check: str, status: str,
                   n_bad: Optional[int], n_total: Optional[int]) -> None:
        """Add a check result to the results list."""
        self.results.append({
            "table": table,
            "check": check,
            "status": status,
            "n_bad": int(n_bad) if n_bad is not None else None,
            "n_total": int(n_total) if n_total is not None else None,
            "pct_bad": self.calculate_percentage(n_bad, n_total)
            if (n_bad is not None and n_total) else None
        })

    def check_duplicates(self, table_name: str, df: pd.DataFrame) -> None:
        """Check for duplicate rows across all columns."""
        if df is None:
            self.add_result(table_name, "duplicates_all_cols", "skip", None, None)
            return

        duplicates = df.duplicated().sum()
        status = "ok" if duplicates == 0 else "warn"
        self.add_result(table_name, "duplicates_all_cols", status, duplicates, len(df))

    def check_primary_key_uniqueness(self, table_name: str, df: pd.DataFrame) -> None:
        """Check if primary key values are unique."""
        if df is None:
            key = self.get_primary_key(table_name)
            self.add_result(table_name, f"unique({key})", "skip", None, None)
            return

        key = self.get_primary_key(table_name)

        if key not in df.columns:
            self.add_result(table_name, f"unique({key})", "skip", None, None)
            return

        duplicate_keys = df[key].duplicated().sum()
        status = "ok" if duplicate_keys == 0 else "warn"
        self.add_result(table_name, f"unique({key})", status, duplicate_keys, len(df))

    def check_table_integrity(self, table_name: str) -> None:
        """Run all integrity checks for a single table."""
        df = self.tables.get(table_name)
        self.check_duplicates(table_name, df)
        self.check_primary_key_uniqueness(table_name, df)

    def check_foreign_key(self, child_table: str, child_key: str,
                          parent_table: str, parent_key: str,
                          label: str) -> None:
        """Check foreign key constraint between two tables."""
        child_df = self.tables.get(child_table)
        parent_df = self.tables.get(parent_table)

        # Skip if tables or columns don't exist
        if (child_df is None or parent_df is None or
                child_key not in child_df.columns or
                parent_key not in parent_df.columns):
            self.add_result(label, f"fk({child_key}->{parent_key})", "skip", None, None)
            return

        total = len(child_df)
        missing = (~child_df[child_key].isin(parent_df[parent_key])).sum()
        status = "ok" if missing == 0 else "warn"
        self.add_result(label, f"fk({child_key}->{parent_key})", status, missing, total)

    def check_foreign_key_nullable(self, child_table: str, child_key: str,
                                   parent_table: str, parent_key: str,
                                   label: str) -> None:
        """Check foreign key constraint, excluding NULL values."""
        child_df = self.tables.get(child_table)
        parent_df = self.tables.get(parent_table)

        if (child_df is None or parent_df is None or
                child_key not in child_df.columns or
                parent_key not in parent_df.columns):
            self.add_result(label, f"fk({child_key}->{parent_key})", "skip", None, None)
            return

        non_null = child_df[child_key].notna()
        total = non_null.sum()

        if total == 0:
            self.add_result(label, f"fk({child_key}->{parent_key})", "ok", 0, 0)
            return

        missing = (~child_df.loc[non_null, child_key].isin(parent_df[parent_key])).sum()
        status = "ok" if missing == 0 else "warn"
        self.add_result(label, f"fk({child_key}->{parent_key})", status, missing, total)

    def check_all_foreign_keys(self) -> None:
        """Check all foreign key relationships in the schema."""
        # r_track_artist relationships
        self.check_foreign_key("r_track_artist", "track_id",
                               "tracks", self.get_primary_key("tracks"),
                               "r_track_artist:tracks")
        self.check_foreign_key("r_track_artist", "artist_id",
                               "artists", self.get_primary_key("artists"),
                               "r_track_artist:artists")

        # r_albums_tracks relationships
        self.check_foreign_key("r_albums_tracks", "track_id",
                               "tracks", self.get_primary_key("tracks"),
                               "r_albums_tracks:tracks")
        self.check_foreign_key("r_albums_tracks", "album_id",
                               "albums", self.get_primary_key("albums"),
                               "r_albums_tracks:albums")

        # tracks.audio_feature_id -> audio_features.id (nullable)
        self.check_foreign_key_nullable("tracks", "audio_feature_id",
                                        "audio_features", "id",
                                        "tracks")

        # r_artist_genre relationships
        self.check_foreign_key("r_artist_genre", "artist_id",
                               "artists", self.get_primary_key("artists"),
                               "r_artist_genre:artists")
        self.check_foreign_key("r_artist_genre", "genre_id",
                               "genres", self.get_primary_key("genres"),
                               "r_artist_genre:genres")

    def run_all_checks(self) -> pd.DataFrame:
        """Run all integrity checks and return results as DataFrame."""
        self.results = []  # Reset results

        # Check individual tables
        for table_name in ["tracks", "audio_features", "artists", "albums", "genres"]:
            self.check_table_integrity(table_name)

        # Check foreign keys
        self.check_all_foreign_keys()

        # Create results DataFrame
        report = pd.DataFrame(self.results).sort_values(["table", "check"]).reset_index(drop=True)
        return report

    def save_report(self, report: pd.DataFrame) -> Path:
        """Save the integrity report to CSV."""
        output_path = self.reports_dir / "integrity_report.csv"
        report.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    def execute(self, display_report: bool = True) -> Tuple[pd.DataFrame, Path]:
        """
        Execute the full integrity check workflow.

        Args:
            display_report: Whether to display the report (for Jupyter notebooks)

        Returns:
            Tuple of (report DataFrame, output file path)
        """
        self.load_all_tables()
        report = self.run_all_checks()

        if display_report:
            try:
                from IPython.display import display
                display(report)
            except ImportError:
                print(report)

        output_path = self.save_report(report)
        print(f"✓ Integrity report saved to: {output_path.resolve()}")

        return report, output_path


# Usage example (replace with your actual paths):
if __name__ == "__main__":
    # checker = DataIntegrityChecker(
    #     data_dir=DATA_DIR,
    #     schema_reports_dir=SCHEMA_REPORTS_DIR
    # )
    # report, path = checker.execute()
    pass