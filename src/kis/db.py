"""SQLite source of truth for videos, shots and keyframes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    pack_id     TEXT,
    video_path  TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    width       INTEGER,
    height      INTEGER,
    codec       TEXT,
    fps_source  REAL,
    time_base   TEXT,
    nb_frames   INTEGER,
    vfr         INTEGER NOT NULL DEFAULT 0,
    degraded    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shots (
    video_id    TEXT NOT NULL,
    shot_id     INTEGER NOT NULL,
    t_start_ms  INTEGER NOT NULL,
    t_end_ms    INTEGER NOT NULL,
    start_frame INTEGER,
    PRIMARY KEY (video_id, shot_id)
);

CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT NOT NULL,
    kf_n        INTEGER NOT NULL,
    frame_idx   INTEGER,
    t_ms        INTEGER NOT NULL,
    pts         REAL,
    shot_id     INTEGER,
    image_path  TEXT,
    thumb_path  TEXT,
    source      TEXT,
    phash       INTEGER,
    UNIQUE (video_id, kf_n),
    UNIQUE (video_id, t_ms)
);

CREATE INDEX IF NOT EXISTS idx_keyframes_video ON keyframes (video_id, t_ms);
CREATE INDEX IF NOT EXISTS idx_shots_video ON shots (video_id, shot_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_video(conn: sqlite3.Connection, video: dict) -> None:
    conn.execute(
        """
        INSERT INTO videos (
            video_id, pack_id, video_path, duration_ms, width, height, codec,
            fps_source, time_base, nb_frames, vfr
        ) VALUES (
            :video_id, :pack_id, :video_path, :duration_ms, :width, :height, :codec,
            :fps_source, :time_base, :nb_frames, :vfr
        )
        ON CONFLICT(video_id) DO UPDATE SET
            pack_id     = excluded.pack_id,
            video_path  = excluded.video_path,
            duration_ms = excluded.duration_ms,
            width       = excluded.width,
            height      = excluded.height,
            codec       = excluded.codec,
            fps_source  = excluded.fps_source,
            time_base   = excluded.time_base,
            nb_frames   = excluded.nb_frames,
            vfr         = excluded.vfr
        """,
        video,
    )


def get_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM videos ORDER BY video_id"))


def get_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()


def get_shots(conn: sqlite3.Connection, video_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT shot_id, t_start_ms, t_end_ms, start_frame FROM shots"
        " WHERE video_id = ? ORDER BY shot_id",
        (video_id,),
    )
    return [dict(r) for r in rows]


def count_keyframes(conn: sqlite3.Connection, video_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM keyframes WHERE video_id = ?", (video_id,)
    ).fetchone()[0]


def replace_shots(conn: sqlite3.Connection, video_id: str, shots: Iterable[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM shots WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO shots (video_id, shot_id, t_start_ms, t_end_ms, start_frame)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (video_id, s["shot_id"], s["t_start_ms"], s["t_end_ms"], s.get("start_frame"))
                for s in shots
            ],
        )


def set_degraded(conn: sqlite3.Connection, video_id: str, degraded: bool) -> None:
    conn.execute(
        "UPDATE videos SET degraded = ? WHERE video_id = ?",
        (1 if degraded else 0, video_id),
    )
    conn.commit()


def replace_keyframes(conn: sqlite3.Connection, video_id: str, rows: list[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM keyframes WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO keyframes (
                video_id, kf_n, frame_idx, t_ms, pts, shot_id,
                image_path, thumb_path, source, phash
            ) VALUES (
                :video_id, :kf_n, :frame_idx, :t_ms, :pts, :shot_id,
                :image_path, :thumb_path, :source, :phash
            )
            """,
            rows,
        )


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
