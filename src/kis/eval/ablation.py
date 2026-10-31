"""Candidate-level ablation across sampling profiles V0-V3.

No frames are decoded here: the candidate timestamps selected from the real
shot boundaries and I-frame pool are measured directly. This is the cheap
"preview" that ranks profiles before spending CPU on extraction.
"""

from __future__ import annotations

from kis import db
from kis.eval.coverage import _covered_ms, _percentile
from kis.ingest import candidates as candidates_mod

PROFILES = ("v0", "v1", "v2", "v3")
PROFILE_NAMES = {
    "v0": "I-frame only",
    "v1": "1 fps grid",
    "v2": "shot midpoint",
    "v3": "hybrid (default)",
}


def _gaps(times: list[int], duration_ms: int) -> list[int]:
    if not times:
        return [duration_ms]
    return [times[0], *[b - a for a, b in zip(times, times[1:])], max(0, duration_ms - times[-1])]


def compute_ablation(
    conn,
    *,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    cfg: dict | None = None,
    radius_ms: int = 1000,
    limit: int | None = None,
) -> dict:
    videos = db.get_videos(conn)
    if limit:
        videos = videos[:limit]

    profiles = {
        p: {
            "n_videos": 0,
            "n_candidates": 0,
            "covered_ms": 0,
            "duration_ms": 0,
            "gaps": [],
            "degraded": 0,
            "by_source": {},
            "per_video": [],
        }
        for p in PROFILES
    }

    for video in videos:
        video_id = video["video_id"]
        shots = db.get_shots(conn, video_id)
        if not shots:
            continue
        duration_ms = int(video["duration_ms"] or 0)
        if duration_ms <= 0:
            continue
        iframes = candidates_mod.iframe_timestamps(video["video_path"], ffprobe, ffmpeg)

        for profile in PROFILES:
            picks, degraded = candidates_mod.select_candidates(
                shots, iframes, duration_ms, profile=profile, cfg=cfg
            )
            times = sorted(c["target_ms"] for c in picks)
            gaps = _gaps(times, duration_ms)
            covered = _covered_ms(times, duration_ms, radius_ms)
            bucket = profiles[profile]
            bucket["n_videos"] += 1
            bucket["n_candidates"] += len(times)
            bucket["covered_ms"] += covered
            bucket["duration_ms"] += duration_ms
            bucket["gaps"].extend(gaps)
            bucket["degraded"] += int(degraded)
            for pick in picks:
                source = pick["source"]
                bucket["by_source"][source] = bucket["by_source"].get(source, 0) + 1
            bucket["per_video"].append(
                {
                    "video_id": video_id,
                    "n_candidates": len(times),
                    "coverage": (covered / duration_ms) if duration_ms else 0.0,
                    "max_gap_ms": max(gaps) if gaps else duration_ms,
                    "degraded": bool(degraded),
                }
            )

    summary = {}
    for profile, bucket in profiles.items():
        gaps = bucket["gaps"]
        summary[profile] = {
            "n_videos": bucket["n_videos"],
            "n_candidates": bucket["n_candidates"],
            "coverage_1s": (bucket["covered_ms"] / bucket["duration_ms"])
            if bucket["duration_ms"]
            else 0.0,
            "max_gap_p50": _percentile(gaps, 50),
            "max_gap_p95": _percentile(gaps, 95),
            "max_gap_p99": _percentile(gaps, 99),
            "max_gap_max": max(gaps) if gaps else 0,
            "degraded": bucket["degraded"],
            "by_source": bucket["by_source"],
            "per_video": bucket["per_video"],
        }
    return summary


def gate_passes(row: dict) -> bool:
    return row["max_gap_p99"] <= 3000 and row["coverage_1s"] >= 0.95


def render_report(summary: dict) -> str:
    lines = [
        "# Keyframe profile ablation (candidate level)",
        "",
        "Measured on the candidate timestamps before decoding; the extracted",
        "keyframe count can be lower after min-gap / pHash dedup.",
        "",
        "| Profile | Strategy | Videos | Candidates | Coverage ±1 s | "
        "max gap p50 / p95 / p99 / max (ms) | Degraded | Gate | Sources |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for profile in PROFILES:
        row = summary[profile]
        sources = ", ".join(f"{k}={v}" for k, v in sorted(row["by_source"].items())) or "-"
        status = "PASS" if gate_passes(row) else "FAIL"
        lines.append(
            f"| {profile} | {PROFILE_NAMES[profile]} | {row['n_videos']} | "
            f"{row['n_candidates']} | {row['coverage_1s']:.4f} | "
            f"{row['max_gap_p50']} / {row['max_gap_p95']} / {row['max_gap_p99']} / "
            f"{row['max_gap_max']} | {row['degraded']} | {status} | {sources} |"
        )
    return "\n".join(lines) + "\n"
