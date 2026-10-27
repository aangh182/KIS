"""K4/K5: PTS-accurate frame extraction, JPEG/thumbnail writing, pHash dedup."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import av
import imagehash
from PIL import Image

from kis import db
from kis.ingest import candidates as candidates_mod
from kis.ingest.dedup import is_near_duplicate, phash_to_int


def _fit_long_side(img: Image.Image, long_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= long_side:
        return img
    scale = long_side / longest
    return img.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def extract_frames(
    video_path: str | Path,
    candidates: list[dict],
    *,
    video_id: str,
    pack_id: str,
    index_dir: Path,
    keyframes_dir: Path,
    thumbnails_dir: Path,
    cfg: dict,
) -> tuple[list[dict], dict]:
    keyframes_dir = Path(keyframes_dir)
    thumbnails_dir = Path(thumbnails_dir)
    index_dir = Path(index_dir)
    min_gap_ms = int(cfg.get("min_gap_ms", 800))
    phash_max_gap_ms = int(cfg.get("phash_max_gap_ms", 2000))
    kf_side = int(cfg.get("keyframe_long_side", 384))
    thumb_side = int(cfg.get("thumb_long_side", 224))
    jpeg_quality = int(cfg.get("jpeg_quality", 85))
    thumb_quality = int(cfg.get("thumb_quality", 80))

    targets = sorted(candidates, key=lambda c: c["target_ms"])
    out_dir = keyframes_dir / pack_id / video_id
    thumb_dir = thumbnails_dir / pack_id / video_id

    rows: list[dict] = []
    dropped_gap = 0
    dropped_phash = 0
    ti = 0
    frame_idx = 0

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)

        for frame in container.decode(stream):
            idx = frame_idx
            frame_idx += 1
            if frame.pts is None:
                continue
            ts_ms = float(frame.pts) * time_base * 1000.0

            matched = []
            while ti < len(targets) and targets[ti]["target_ms"] <= ts_ms + 0.5:
                matched.append(targets[ti])
                ti += 1

            if matched:
                t_ms = int(round(ts_ms))
                candidate = matched[0]
                shot_id = candidate.get("shot_id")

                if rows and t_ms - rows[-1]["t_ms"] < min_gap_ms:
                    dropped_gap += 1
                else:
                    img = frame.to_image()
                    phash = phash_to_int(imagehash.phash(img.convert("L")))

                    duplicate = False
                    if shot_id is not None:
                        for prev in reversed(rows[-6:]):
                            prev_shot = prev.get("shot_id")
                            if prev_shot is None:
                                continue
                            if abs(prev_shot - shot_id) > 2:
                                break
                            if is_near_duplicate(phash, prev["phash"]):
                                duplicate = True
                                break

                    if duplicate:
                        next_target = targets[ti]["target_ms"] if ti < len(targets) else None
                        if (
                            not rows
                            or next_target is None
                            or next_target - rows[-1]["t_ms"] > phash_max_gap_ms
                        ):
                            duplicate = False

                    if duplicate:
                        dropped_phash += 1
                    else:
                        kf_n = len(rows) + 1
                        out_dir.mkdir(parents=True, exist_ok=True)
                        thumb_dir.mkdir(parents=True, exist_ok=True)

                        keyframe_img = _fit_long_side(img, kf_side)
                        thumb_img = _fit_long_side(img, thumb_side)
                        kf_path = out_dir / f"{kf_n}.jpg"
                        thumb_path = thumb_dir / f"{kf_n}.jpg"
                        keyframe_img.save(kf_path, "JPEG", quality=jpeg_quality)
                        thumb_img.save(thumb_path, "JPEG", quality=thumb_quality)

                        rows.append(
                            {
                                "video_id": video_id,
                                "kf_n": kf_n,
                                "frame_idx": idx,
                                "t_ms": t_ms,
                                "pts": ts_ms / 1000.0,
                                "shot_id": shot_id,
                                "image_path": str(kf_path.relative_to(index_dir)),
                                "thumb_path": str(thumb_path.relative_to(index_dir)),
                                "source": candidate.get("source", "unknown"),
                                "phash": phash,
                            }
                        )
                    img.close()

            if ti >= len(targets):
                break
    finally:
        container.close()

    stats = {
        "n_candidates": len(targets),
        "n_keyframes": len(rows),
        "dropped_gap": dropped_gap,
        "dropped_phash": dropped_phash,
    }
    return rows, stats


def process_video(job: dict) -> dict:
    """Worker entry point: decode, dedup, write JPEGs, persist rows."""
    started = time.time()
    conn = db.connect(job["db_path"])
    try:
        db.migrate(conn)
        video_id = job["video_id"]
        if not job.get("force") and db.count_keyframes(conn, video_id) > 0:
            return {"video_id": video_id, "skipped": True, "seconds": 0.0}

        video = db.get_video(conn, video_id)
        if video is None:
            return {"video_id": video_id, "error": "video not in database"}

        shots = db.get_shots(conn, video_id)
        if not shots:
            return {"video_id": video_id, "error": "no shots; run `kis shots` first"}

        cfg = job["cfg"]
        iframes = candidates_mod.iframe_timestamps(
            video["video_path"], job.get("ffprobe", "ffprobe"), job.get("ffmpeg", "ffmpeg")
        )

        picks, degraded = candidates_mod.select_candidates(
            shots,
            iframes,
            int(video["duration_ms"]),
            profile=job["profile"],
            cfg=cfg,
        )

        index_dir = Path(job["index_dir"])
        pack_id = video["pack_id"] or video_id.split("_", 1)[0]
        out_dir = Path(job["keyframes_dir"]) / pack_id / video_id
        thumb_dir = Path(job["thumbnails_dir"]) / pack_id / video_id
        if job.get("force"):
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(thumb_dir, ignore_errors=True)

        rows, stats = extract_frames(
            video["video_path"],
            picks,
            video_id=video_id,
            pack_id=pack_id,
            index_dir=index_dir,
            keyframes_dir=Path(job["keyframes_dir"]),
            thumbnails_dir=Path(job["thumbnails_dir"]),
            cfg=cfg,
        )
        db.replace_keyframes(conn, video_id, rows)
        db.set_degraded(conn, video_id, degraded)

        return {
            "video_id": video_id,
            "skipped": False,
            "degraded": degraded,
            "n_shots": len(shots),
            "n_iframes": len(iframes),
            **stats,
            "seconds": time.time() - started,
        }
    except Exception as exc:  # noqa: BLE001 - surface per-video failure to the pool
        return {"video_id": job.get("video_id"), "error": str(exc), "seconds": time.time() - started}
    finally:
        conn.close()
