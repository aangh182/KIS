"""GT-free keyframe coverage metrics: temporal coverage and gap distribution."""

from __future__ import annotations

import math

from kis import db


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _covered_ms(times: list[int], duration_ms: int, radius_ms: int) -> int:
    if not times or duration_ms <= 0:
        return 0
    total = 0
    cur_start = None
    cur_end = None
    for t in times:
        start = max(0, t - radius_ms)
        end = min(duration_ms, t + radius_ms)
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def compute_coverage(conn, radius_ms: int = 1000) -> dict:
    videos = db.get_videos(conn)
    per_video = []
    all_gaps: list[int] = []
    covered_total = 0
    duration_total = 0
    nkeyframes = 0

    for video in videos:
        video_id = video["video_id"]
        duration_ms = int(video["duration_ms"] or 0)
        times = [
            int(r[0])
            for r in conn.execute(
                "SELECT t_ms FROM keyframes WHERE video_id = ? ORDER BY t_ms", (video_id,)
            )
        ]
        nkeyframes += len(times)
        if times:
            gaps = [times[0]]
            gaps.extend(b - a for a, b in zip(times, times[1:]))
            gaps.append(max(0, duration_ms - times[-1]))
        else:
            gaps = [duration_ms]
        all_gaps.extend(gaps)

        covered = _covered_ms(times, duration_ms, radius_ms)
        covered_total += covered
        duration_total += duration_ms
        per_video.append(
            {
                "video_id": video_id,
                "n_keyframes": len(times),
                "duration_ms": duration_ms,
                "max_gap_ms": max(gaps) if gaps else duration_ms,
                "coverage": (covered / duration_ms) if duration_ms else 0.0,
                "degraded": bool(video["degraded"]),
            }
        )

    coverage = (covered_total / duration_total) if duration_total else 0.0
    return {
        "nvideos": len(videos),
        "nkeyframes": nkeyframes,
        "coverage_1s": coverage,
        "max_gap_p50": _percentile(all_gaps, 50),
        "max_gap_p95": _percentile(all_gaps, 95),
        "max_gap_p99": _percentile(all_gaps, 99),
        "max_gap_max": max(all_gaps) if all_gaps else 0,
        "per_video": per_video,
    }


def gate_passes(summary: dict) -> bool:
    return summary["max_gap_p99"] <= 3000 and summary["coverage_1s"] >= 0.95


def render_report(summary: dict) -> str:
    status = "PASS" if gate_passes(summary) else "FAIL"
    lines = [
        "# Keyframe coverage report",
        "",
        f"Gate (max_gap p99 <= 3000 ms, coverage >= 0.95): **{status}**",
        "",
        f"- videos: {summary['nvideos']}",
        f"- keyframes: {summary['nkeyframes']}",
        f"- coverage (±1 s): {summary['coverage_1s']:.4f}",
        f"- max gap p50/p95/p99/max: "
        f"{summary['max_gap_p50']} / {summary['max_gap_p95']} / "
        f"{summary['max_gap_p99']} / {summary['max_gap_max']} ms",
        "",
        "| video | keyframes | degraded | max gap (ms) | coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_video"]:
        lines.append(
            f"| {row['video_id']} | {row['n_keyframes']} | "
            f"{'yes' if row['degraded'] else 'no'} | {row['max_gap_ms']} | "
            f"{row['coverage']:.4f} |"
        )
    return "\n".join(lines) + "\n"
