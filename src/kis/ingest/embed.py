"""SigLIP-2 image tower: keyframe JPEGs -> normalized fp16 .npy shards."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kis import db


def shard_paths(embeddings_dir: Path, pack_id: str, video_id: str) -> tuple[Path, Path]:
    base = Path(embeddings_dir) / pack_id
    return base / f"{video_id}.npy", base / f"{video_id}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sidecar(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def shard_is_valid(
    npy_path: Path, meta_path: Path, model_id: str, keyframe_pks: list[int]
) -> bool:
    meta = load_sidecar(meta_path)
    if meta is None or not npy_path.exists():
        return False
    if meta.get("model") != model_id:
        return False
    if meta.get("keyframe_pks") != keyframe_pks:
        return False
    if int(meta.get("count", -1)) != len(keyframe_pks):
        return False
    return meta.get("sha256") == sha256_file(npy_path)


def load_model(model_id: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _load_images(index_dir: Path, rows: list[dict]) -> list:
    from PIL import Image

    images = []
    for row in rows:
        path = index_dir / row["image_path"]
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _feature_tensor(output):
    # transformers >=5 wraps pooled features in BaseModelOutputWithPooling.
    return output.pooler_output if hasattr(output, "pooler_output") else output


def embed_rows(
    model,
    processor,
    index_dir: Path,
    rows: list[dict],
    batch_size: int,
    dtype: str = "float16",
    on_batch=None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    chunks: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = _load_images(index_dir, batch)
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            features = _feature_tensor(model.get_image_features(**inputs))
        features = functional.normalize(features.float(), p=2, dim=-1)
        array = features.cpu().numpy().astype(np.dtype(dtype))
        chunks.append(array)
        for image in images:
            image.close()
        if on_batch is not None:
            on_batch(len(batch))

    if not chunks:
        return np.empty((0, 0), dtype=np.dtype(dtype))
    return np.concatenate(chunks, axis=0)


def write_shard(
    npy_path: Path,
    meta_path: Path,
    array: np.ndarray,
    *,
    video_id: str,
    model_id: str,
    keyframe_pks: list[int],
    dtype: str,
) -> dict:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save appends ".npy" unless the path already ends with it.
    tmp_npy = npy_path.with_suffix(".tmp.npy")
    np.save(tmp_npy, array)
    os.replace(tmp_npy, npy_path)

    meta = {
        "video_id": video_id,
        "model": model_id,
        "dim": int(array.shape[1]) if array.ndim == 2 else 0,
        "dtype": dtype,
        "count": int(array.shape[0]),
        "keyframe_pks": keyframe_pks,
        "sha256": sha256_file(npy_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return meta


def run_embed(
    db_path: str | Path,
    index_dir: str | Path,
    embeddings_dir: str | Path,
    *,
    model_id: str,
    batch_size: int = 32,
    dtype: str = "float16",
    limit: int | None = None,
    video_id: str | None = None,
    force: bool = False,
) -> dict:
    index_dir = Path(index_dir)
    embeddings_dir = Path(embeddings_dir)
    conn = db.connect(db_path)
    db.migrate(conn)
    try:
        videos = db.get_videos(conn)
        if video_id:
            videos = [v for v in videos if v["video_id"] == video_id]
        if limit:
            videos = videos[:limit]

        pending = []
        for video in videos:
            rows = db.get_keyframes_for_video(conn, video["video_id"])
            if rows:
                pending.append((video, rows))

        if not pending:
            return {"videos": 0, "images": 0, "skipped": 0, "seconds": 0.0, "dim": 0}

        jobs = []
        skipped = 0
        for video, rows in pending:
            pack_id = video["pack_id"] or video["video_id"].split("_", 1)[0]
            keyframe_pks = [int(row["keyframe_pk"]) for row in rows]
            npy_path, meta_path = shard_paths(embeddings_dir, pack_id, video["video_id"])
            if not force and shard_is_valid(npy_path, meta_path, model_id, keyframe_pks):
                skipped += 1
                continue
            jobs.append((video, rows, keyframe_pks, npy_path, meta_path))

        total_images = sum(len(rows) for _, rows, *_ in jobs)
        if not jobs:
            return {
                "videos": 0,
                "images": 0,
                "skipped": skipped,
                "seconds": 0.0,
                "dim": 0,
            }

        model, processor = load_model(model_id)
        started = time.time()
        images_done = 0
        embedded_videos = 0
        dim = 0

        from tqdm import tqdm

        with tqdm(total=total_images, unit="img", desc="embed") as bar:
            for video, rows, keyframe_pks, npy_path, meta_path in jobs:
                array = embed_rows(
                    model,
                    processor,
                    index_dir,
                    rows,
                    batch_size,
                    dtype,
                    on_batch=bar.update,
                )
                if array.size == 0:
                    continue
                write_shard(
                    npy_path,
                    meta_path,
                    array,
                    video_id=video["video_id"],
                    model_id=model_id,
                    keyframe_pks=keyframe_pks,
                    dtype=dtype,
                )
                dim = int(array.shape[1])
                images_done += int(array.shape[0])
                embedded_videos += 1

        seconds = time.time() - started
        return {
            "videos": embedded_videos,
            "images": images_done,
            "skipped": skipped,
            "seconds": seconds,
            "img_per_s": (images_done / seconds) if seconds > 0 else 0.0,
            "dim": dim,
        }
    finally:
        conn.close()
