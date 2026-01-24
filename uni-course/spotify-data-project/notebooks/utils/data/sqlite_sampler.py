from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import sqlite3
import pandas as pd


@dataclass
class SamplerReport:
    sample_name: str
    mode: str
    target_tracks: int
    selected_tracks: int
    row_buckets: Optional[List[int]]
    explicit_count: int
    hit_like_count: int
    avg_popularity: float
    export_files: Dict[str, str]
    elapsed_sec: float


class SQLiteSampleExporter:
    """
    Exports an ML-friendly connected subgraph sample from a big SQLite DB
    without loading full tables into pandas.
    """

    def __init__(
        self,
        db_path: Path,
        export_dir: Path,
        sample_name: str,
        require_audio_features: bool = True,
    ):
        self.db_path = Path(db_path)
        self.export_dir = Path(export_dir)
        self.sample_name = sample_name
        self.require_audio_features = require_audio_features

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        return con

    def _apply_pragmas(self, con: sqlite3.Connection) -> None:
        con.execute("PRAGMA journal_mode=OFF;")
        con.execute("PRAGMA synchronous=OFF;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=30000000000;")

    def _cleanup_temp(self, cur: sqlite3.Cursor) -> None:
        cur.executescript("""
        DROP TABLE IF EXISTS temp.sel_tracks;
        DROP TABLE IF EXISTS temp.sel_r_albums_tracks;
        DROP TABLE IF EXISTS temp.sel_r_track_artist;
        DROP TABLE IF EXISTS temp.sel_albums;
        DROP TABLE IF EXISTS temp.sel_artists;
        DROP TABLE IF EXISTS temp.sel_audio_features;
        DROP TABLE IF EXISTS temp.sel_r_artist_genre;
        DROP TABLE IF EXISTS temp.sel_genres;
        DROP TABLE IF EXISTS temp.sel_r_albums_artists;
        """)

    def export_rowid_slice(
        self,
        *,
        target_tracks: int,
        row_mod: int,
        row_k_start: int,
        buckets_per_slice: int,
        limit: Optional[int] = None,
    ) -> SamplerReport:
        """
        Deterministic slicing: selects tracks by (rowid % row_mod) in a bucket set.
        Great for reproducible multi-run evaluation across slices.
        """
        t0 = time.time()
        self.export_dir.mkdir(parents=True, exist_ok=True)

        audio_filter = "AND audio_feature_id IS NOT NULL" if self.require_audio_features else ""
        row_ks = [(row_k_start + i) % row_mod for i in range(buckets_per_slice)]
        row_k_sql = ",".join(map(str, row_ks))
        hard_limit = limit or target_tracks

        with self._connect() as con:
            self._apply_pragmas(con)
            cur = con.cursor()
            self._cleanup_temp(cur)

            # 1) Select tracks
            cur.execute(f"""
                CREATE TEMP TABLE sel_tracks AS
                SELECT id AS track_id,
                       disc_number, duration, explicit, audio_feature_id,
                       name, preview_url, track_number, popularity, is_playable
                FROM tracks
                WHERE (rowid % {row_mod}) IN ({row_k_sql})
                  {audio_filter}
                  AND duration IS NOT NULL AND duration > 0
                  AND popularity IS NOT NULL
                LIMIT {hard_limit};
            """)
            cur.executescript("""
                CREATE INDEX IF NOT EXISTS idx_sel_tracks_id ON sel_tracks(track_id);
                CREATE INDEX IF NOT EXISTS idx_sel_tracks_af ON sel_tracks(audio_feature_id);
            """)

            n_tracks = cur.execute("SELECT COUNT(*) FROM sel_tracks;").fetchone()[0]

            stats = cur.execute("""
            SELECT
              SUM(CASE WHEN explicit=1 THEN 1 ELSE 0 END) AS n_explicit,
              SUM(CASE WHEN popularity>=70 THEN 1 ELSE 0 END) AS n_hit_like,
              AVG(popularity) AS avg_pop
            FROM sel_tracks;
            """).fetchone()
            n_explicit, n_hit_like, avg_pop = int(stats[0] or 0), int(stats[1] or 0), float(stats[2] or 0.0)

            # 2) Relations
            cur.executescript("""
            CREATE TEMP TABLE sel_r_albums_tracks AS
            SELECT r.album_id, r.track_id
            FROM r_albums_tracks r
            JOIN sel_tracks t ON t.track_id = r.track_id;

            CREATE INDEX IF NOT EXISTS idx_sel_rat_track ON sel_r_albums_tracks(track_id);
            CREATE INDEX IF NOT EXISTS idx_sel_rat_album ON sel_r_albums_tracks(album_id);

            CREATE TEMP TABLE sel_r_track_artist AS
            SELECT r.track_id, r.artist_id
            FROM r_track_artist r
            JOIN sel_tracks t ON t.track_id = r.track_id;

            CREATE INDEX IF NOT EXISTS idx_sel_rta_track ON sel_r_track_artist(track_id);
            CREATE INDEX IF NOT EXISTS idx_sel_rta_artist ON sel_r_track_artist(artist_id);
            """)

            # 3) Entities
            cur.executescript("""
            CREATE TEMP TABLE sel_albums AS
            SELECT a.id, a.name, a.album_group, a.album_type, a.release_date, a.popularity
            FROM albums a
            JOIN sel_r_albums_tracks rat ON rat.album_id = a.id
            GROUP BY a.id;

            CREATE INDEX IF NOT EXISTS idx_sel_albums_id ON sel_albums(id);

            CREATE TEMP TABLE sel_artists AS
            SELECT ar.id, ar.name, ar.popularity, ar.followers
            FROM artists ar
            JOIN sel_r_track_artist rta ON rta.artist_id = ar.id
            GROUP BY ar.id;

            CREATE INDEX IF NOT EXISTS idx_sel_artists_id ON sel_artists(id);

            CREATE TEMP TABLE sel_audio_features AS
            SELECT af.id, af.acousticness, af.analysis_url, af.danceability, af.duration,
                   af.energy, af.instrumentalness, af.key, af.liveness, af.loudness,
                   af.mode, af.speechiness, af.tempo, af.time_signature, af.valence
            FROM audio_features af
            JOIN sel_tracks t ON t.audio_feature_id = af.id
            GROUP BY af.id;

            CREATE INDEX IF NOT EXISTS idx_sel_af_id ON sel_audio_features(id);
            """)

            cur.executescript("""
            CREATE TEMP TABLE sel_r_artist_genre AS
            SELECT rag.genre_id, rag.artist_id
            FROM r_artist_genre rag
            JOIN sel_artists a ON a.id = rag.artist_id;

            CREATE INDEX IF NOT EXISTS idx_sel_rag_artist ON sel_r_artist_genre(artist_id);
            CREATE INDEX IF NOT EXISTS idx_sel_rag_genre  ON sel_r_artist_genre(genre_id);

            CREATE TEMP TABLE sel_genres AS
            SELECT g.id
            FROM genres g
            JOIN sel_r_artist_genre rag ON rag.genre_id = g.id
            GROUP BY g.id;

            CREATE INDEX IF NOT EXISTS idx_sel_genres_id ON sel_genres(id);

            CREATE TEMP TABLE sel_r_albums_artists AS
            SELECT r.album_id, r.artist_id
            FROM r_albums_artists r
            JOIN sel_albums a ON a.id = r.album_id;

            CREATE INDEX IF NOT EXISTS idx_sel_raa_album ON sel_r_albums_artists(album_id);
            CREATE INDEX IF NOT EXISTS idx_sel_raa_artist ON sel_r_albums_artists(artist_id);
            """)

            # 4) Export helper
            def dump(name: str, sql: str) -> Path:
                df = pd.read_sql(sql, con)
                out = self.export_dir / f"{name}.csv"
                df.to_csv(out, index=False, header=True, encoding="utf-8-sig")
                return out

            export_files = {}
            export_files["tracks"] = dump("tracks", "SELECT * FROM sel_tracks;").name
            export_files["r_albums_tracks"] = dump("r_albums_tracks", "SELECT * FROM sel_r_albums_tracks;").name
            export_files["r_track_artist"] = dump("r_track_artist", "SELECT * FROM sel_r_track_artist;").name
            export_files["albums"] = dump("albums", "SELECT * FROM sel_albums;").name
            export_files["artists"] = dump("artists", "SELECT * FROM sel_artists;").name
            export_files["audio_features"] = dump("audio_features", "SELECT * FROM sel_audio_features;").name
            export_files["r_artist_genre"] = dump("r_artist_genre", "SELECT * FROM sel_r_artist_genre;").name
            export_files["genres"] = dump("genres", "SELECT * FROM sel_genres;").name
            export_files["r_albums_artists"] = dump("r_albums_artists", "SELECT * FROM sel_r_albums_artists;").name

            # track_id list
            df_ids = pd.read_sql("SELECT track_id FROM sel_tracks;", con)
            df_ids.to_csv(self.export_dir / "selected_track_ids.csv", index=False)

        elapsed = time.time() - t0
        return SamplerReport(
            sample_name=self.sample_name,
            mode="ROWID_MOD",
            target_tracks=target_tracks,
            selected_tracks=n_tracks,
            row_buckets=row_ks,
            explicit_count=n_explicit,
            hit_like_count=n_hit_like,
            avg_popularity=avg_pop,
            export_files=export_files,
            elapsed_sec=elapsed,
        )