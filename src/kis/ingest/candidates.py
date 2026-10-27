"""K2a/K3: candidate pools and the hybrid selection policy (plan section 3.3)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")

PROFILE_ALIASES = {
    "v0": "v0",
    "iframe": "v0",
    "v1": "v1",
    "grid": "v1",
    "v2": "v2",
    "shot": "v2",
    "v3": "v3",
    "hybrid": "v3",
}


def profile_key(profile: str) -> str:
    key = PROFILE_ALIASES.get(profile.lower())
    if key is None:
        raise ValueError(f"unknown sampling profile: {profile}")
    return key


def iframe_timestamps(
    video_path: str | Path, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg"
) -> list[int]:
    """Demux-only I-frame PTS pool, in milliseconds."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time,key_frame",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        stamps: list[int] = []
        for line in proc.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                key_frame = int(parts[0])
                pts = float(parts[1])
            except ValueError:
                continue
            if key_frame == 1:
                stamps.append(int(round(pts * 1000)))
        if stamps:
            return sorted(set(stamps))

    return _iframes_with_ffmpeg(video_path, ffmpeg)


def _iframes_with_ffmpeg(video_path: str | Path, ffmpeg: str) -> list[int]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-skip_frame",
        "nokey",
        "-i",
        str(video_path),
        "-vf",
        "showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"I-frame pass failed for {video_path}: {proc.stderr[-500:]}")
    return sorted({int(round(float(x) * 1000)) for x in _PTS_RE.findall(proc.stderr)})


def grid_timestamps(duration_ms: int, step_ms: int = 1000, start_ms: int = 0) -> list[int]:
    if duration_ms <= 0 or step_ms <= 0:
        return []
    stamps = []
    t = max(0, start_ms)
    while t < duration_ms:
        stamps.append(t)
        t += step_ms
    return stamps


def _dedup_picks(picks: list[dict], min_gap_ms: int, max_picks: int | None = None) -> list[dict]:
    picks = sorted(picks, key=lambda p: p["target_ms"])
    kept: list[dict] = []
    for pick in picks:
        if kept and pick["target_ms"] - kept[-1]["target_ms"] < min_gap_ms:
            continue
        kept.append(pick)
    if max_picks is not None:
        kept = kept[:max_picks]
    return kept


def _candidate(target_ms: int, source: str, shot_id: int | None = None) -> dict:
    return {"target_ms": int(target_ms), "source": source, "shot_id": shot_id}


def _hybrid_picks(
    shots: list[dict],
    iframes: list[int],
    duration_ms: int,
    cfg: dict,
) -> list[dict]:
    grid_ms = int(cfg["grid_s"] * 1000)
    min_gap_ms = int(cfg["min_gap_ms"])
    max_per_shot = int(cfg["max_per_shot"])

    picks: list[dict] = []
    for shot in shots:
        t0 = int(shot["t_start_ms"])
        t1 = int(shot["t_end_ms"])
        dur = t1 - t0
        shot_id = int(shot["shot_id"])

        inside = [f for f in iframes if t0 <= f <= t1]
        shot_picks = [_candidate(f, "iframe", shot_id) for f in inside]
        if not shot_picks:
            shot_picks = [_candidate((t0 + t1) // 2, "shot", shot_id)]

        # MAX_PER_SHOT bounds semantic picks (I-frames / shot midpoint). The
        # safety grid is not capped, otherwise long shots get holes.
        shot_picks = _dedup_picks(shot_picks, min_gap_ms, max_per_shot)

        if dur > 4000 and grid_ms > 0:
            k = 1
            while t0 + k * grid_ms < t1:
                shot_picks.append(_candidate(t0 + k * grid_ms, "grid", shot_id))
                k += 1

        picks.extend(_dedup_picks(shot_picks, min_gap_ms))
    return picks


def select_candidates(
    shots: list[dict],
    iframes: list[int],
    duration_ms: int,
    profile: str = "hybrid",
    cfg: dict | None = None,
) -> tuple[list[dict], bool]:
    """Return (candidates, degraded).

    A degraded video falls back to the safe 1 fps grid, per plan section 3.3.
    """
    cfg = cfg or {}
    defaults = {
        "grid_s": 3.0,
        "max_per_shot": 6,
        "min_shot_s": 0.4,
        "long_shot_s": 10.0,
        "min_gap_ms": 800,
        "safety_grid_s": 2.0,
    }
    cfg = {**defaults, **cfg}
    key = profile_key(profile)

    if key == "v0":
        picks = [
            _candidate(f, "iframe")
            for f in iframes
            if 0 <= f < duration_ms
        ]
        picks = _dedup_picks(picks, cfg["min_gap_ms"])
        return picks, False

    if key == "v1" or not shots:
        picks = [_candidate(t, "grid") for t in grid_timestamps(duration_ms, 1000)]
        return picks, key == "v3" or not shots

    if key == "v2":
        picks = [_candidate((s["t_start_ms"] + s["t_end_ms"]) // 2, "shot", s["shot_id"]) for s in shots]
        return _dedup_picks(picks, cfg["min_gap_ms"]), False

    long_ms = int(cfg["long_shot_s"] * 1000)
    degraded = len(shots) <= 1 or (
        cfg.get("degrade_on_long_shot", False)
        and any((s["t_end_ms"] - s["t_start_ms"]) > long_ms for s in shots)
    )
    if degraded:
        picks = [_candidate(t, "grid") for t in grid_timestamps(duration_ms, 1000)]
        return picks, True

    picks = _hybrid_picks(shots, iframes, duration_ms, cfg)

    safety_grid_ms = int(cfg.get("safety_grid_s", 0.0) * 1000)
    if safety_grid_ms > 0:
        picks.extend(_candidate(t, "grid") for t in grid_timestamps(duration_ms, safety_grid_ms))
        picks = _dedup_picks(picks, cfg["min_gap_ms"])

    seen: set[int] = set()
    unique: list[dict] = []
    for pick in picks:
        if pick["target_ms"] < 0 or pick["target_ms"] >= duration_ms:
            continue
        if pick["target_ms"] in seen:
            continue
        seen.add(pick["target_ms"])
        unique.append(pick)
    return unique, False


def summarize_candidates(candidates: list[dict]) -> dict:
    by_source: dict[str, int] = {}
    for c in candidates:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    return {"n": len(candidates), "by_source": by_source}


def max_gap_for(candidates: list[dict], duration_ms: int) -> int:
    if not candidates:
        return duration_ms
    times = sorted(c["target_ms"] for c in candidates)
    gaps = [times[0], *[b - a for a, b in zip(times, times[1:])], duration_ms - times[-1]]
    return max(gaps) if gaps else duration_ms
