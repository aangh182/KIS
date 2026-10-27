"""K1: probe video files into the videos table."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kis import db

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
RATIONAL_KEYS = ("r_frame_rate", "avg_frame_rate")


def list_video_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(files)


def _fraction(value: str | None) -> float | None:
    if not value or value in {"N/A", "0/0"}:
        return None
    try:
        num, _, den = value.partition("/")
        den_f = float(den) if den else 1.0
        return float(num) / den_f if den_f else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ffprobe_json(video_path: str | Path, ffprobe: str = "ffprobe") -> dict:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_video(video_path: str | Path, ffprobe: str = "ffprobe") -> dict:
    path = Path(video_path)
    info = ffprobe_json(path, ffprobe)

    stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if stream is None:
        raise RuntimeError(f"no video stream in {path}")

    fmt = info.get("format", {})
    duration_s = _float_or_none(fmt.get("duration"))
    if duration_s is None:
        duration_s = _float_or_none(stream.get("duration"))
    duration_ms = int(round((duration_s or 0.0) * 1000))

    r_fps = _fraction(stream.get("r_frame_rate"))
    avg_fps = _fraction(stream.get("avg_frame_rate"))
    vfr = 0
    if r_fps and avg_fps:
        vfr = int(abs(r_fps - avg_fps) / avg_fps > 0.001)

    video_id = path.stem
    pack_id = video_id.split("_", 1)[0]

    return {
        "video_id": video_id,
        "pack_id": pack_id,
        "video_path": str(path.resolve()),
        "duration_ms": duration_ms,
        "width": _int_or_none(stream.get("width")),
        "height": _int_or_none(stream.get("height")),
        "codec": stream.get("codec_name"),
        "fps_source": avg_fps,
        "time_base": stream.get("time_base"),
        "nb_frames": _int_or_none(stream.get("nb_frames")),
        "vfr": vfr,
    }


def run_probe(db_path: str | Path, dataset_dir: str | Path, ffprobe: str = "ffprobe") -> dict:
    files = list_video_files(dataset_dir)
    conn = db.connect(db_path)
    try:
        db.migrate(conn)
        with conn:
            for path in files:
                db.upsert_video(conn, probe_video(path, ffprobe))
        return {"nvideos": len(files), "total": db.count_rows(conn, "videos")}
    finally:
        conn.close()
