"""Timestamp correctness verification against the ffprobe demuxer."""

from __future__ import annotations

import subprocess


def _frame_pts_ms(video_path: str, t_ms: int, ffprobe: str) -> list[int]:
    start_s = max(0.0, (t_ms - 700) / 1000.0)
    # ffprobe stops the interval relative to the seek keyframe, so the window
    # must be wide enough to include the requested timestamp.
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"{start_s:.3f}%+8.0",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-300:])
    stamps: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            stamps.append(int(round(float(line) * 1000)))
        except ValueError:
            continue
    return stamps


def verify_timestamps(conn, sample: int, ffprobe: str = "ffprobe") -> dict:
    rows = list(
        conn.execute(
            """
            SELECT k.video_id, k.kf_n, k.t_ms, v.video_path
            FROM keyframes k JOIN videos v USING (video_id)
            ORDER BY RANDOM() LIMIT ?
            """,
            (sample,),
        )
    )
    if not rows:
        return {"checked": 0, "passed": 0, "failed": 0, "results": []}

    results = []
    for row in rows:
        try:
            stamps = _frame_pts_ms(row["video_path"], int(row["t_ms"]), ffprobe)
            diff = min((abs(s - int(row["t_ms"])) for s in stamps), default=None)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {"video_id": row["video_id"], "kf_n": row["kf_n"], "t_ms": row["t_ms"], "error": str(exc)}
            )
            continue
        results.append(
            {
                "video_id": row["video_id"],
                "kf_n": row["kf_n"],
                "t_ms": row["t_ms"],
                "diff_ms": diff,
                "ok": diff == 0,
            }
        )

    passed = sum(1 for r in results if r.get("ok"))
    failed = len(results) - passed
    return {"checked": len(results), "passed": passed, "failed": failed, "results": results}
