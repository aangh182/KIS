"""K2b: shot boundary detection.

Primary backend is PySceneDetect's ContentDetector. If it is unavailable we
fall back to ffmpeg's scene score filter, which returns exact demuxer PTS.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def _merge_short(raw: list[dict], min_shot_ms: int) -> list[dict]:
    merged: list[dict] = []
    for shot in raw:
        if merged and (shot["t_end_ms"] - shot["t_start_ms"]) < min_shot_ms:
            merged[-1]["t_end_ms"] = max(merged[-1]["t_end_ms"], shot["t_end_ms"])
        else:
            merged.append(dict(shot))
    for i, shot in enumerate(merged):
        shot["shot_id"] = i
    return merged


def detect_with_scenedetect(
    video_path: str | Path,
    min_shot_ms: int = 400,
    threshold: float = 27.0,
) -> list[dict]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list(start_in_scene=True)

    raw = []
    for start, end in scenes:
        raw.append(
            {
                "t_start_ms": int(round(start.seconds * 1000)),
                "t_end_ms": int(round(end.seconds * 1000)),
                "start_frame": int(start.frame_num),
            }
        )
    if not raw:
        raise RuntimeError(f"PySceneDetect returned no scenes for {video_path}")
    return _merge_short(raw, min_shot_ms)


def detect_with_ffmpeg(
    video_path: str | Path,
    duration_ms: int,
    fps: float | None,
    min_shot_ms: int = 400,
    threshold: float = 0.30,
    ffmpeg: str = "ffmpeg",
) -> list[dict]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        f"scale=-2:360,select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg shot detection failed for {video_path}: {proc.stderr[-500:]}")

    cuts = sorted({int(round(float(x) * 1000)) for x in _PTS_RE.findall(proc.stderr)})
    cuts = [c for c in cuts if 0 < c < duration_ms]
    boundaries = [0, *cuts, duration_ms]

    raw = []
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        start_frame = int(round(start_ms * (fps or 0.0) / 1000)) if fps else None
        raw.append({"t_start_ms": start_ms, "t_end_ms": end_ms, "start_frame": start_frame})
    return _merge_short(raw, min_shot_ms)


def detect_shots(
    video_path: str | Path,
    duration_ms: int,
    fps: float | None,
    backend: str = "auto",
    min_shot_ms: int = 400,
    scene_threshold: float = 0.30,
    ffmpeg: str = "ffmpeg",
) -> tuple[list[dict], str]:
    order = ["scenedetect", "ffmpeg"] if backend in {"auto", "scenedetect"} else ["ffmpeg"]
    if backend == "ffmpeg":
        order = ["ffmpeg"]

    last_error: Exception | None = None
    for name in order:
        try:
            if name == "scenedetect":
                shots = detect_with_scenedetect(video_path, min_shot_ms=min_shot_ms)
            else:
                shots = detect_with_ffmpeg(
                    video_path,
                    duration_ms,
                    fps,
                    min_shot_ms=min_shot_ms,
                    threshold=scene_threshold,
                    ffmpeg=ffmpeg,
                )
            return shots, name
        except Exception as exc:  # noqa: BLE001 - intentional backend fallback
            last_error = exc
            if backend == "scenedetect":
                break
    raise RuntimeError(f"all shot backends failed for {video_path}: {last_error}")
