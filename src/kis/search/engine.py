"""Text query -> SigLIP-2 text tower -> FAISS -> SQLite metadata."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from kis import db
from kis.index.store import FaissStore
from kis.ingest.embed import load_model


def format_timestamp(t_ms: int) -> str:
    total_ms = max(0, int(t_ms))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _media_url(path: str | None) -> str | None:
    if not path:
        return None
    return "/media/" + path.replace("\\", "/").lstrip("/")


def frame_uniformity(path: Path) -> tuple[float, float, float] | None:
    """Return (grayscale std, dominant-color fraction, edge energy).

    A frame is semantically empty when it is nearly uniform (black t=0 frames,
    white flashes) or when one flat color covers most of it (solid-color
    transition wipes). Edge energy separates those wipes from real dark scenes
    and logo screens, which also have a dominant color but carry structure.
    """
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            gray = image.convert("L")
            std = float(ImageStat.Stat(gray).stddev[0])
            values = np.asarray(gray, dtype=np.float32)
            small = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        pixels = np.asarray(small, dtype=np.uint16) // 16
        keys = (pixels[..., 0] << 8) | (pixels[..., 1] << 4) | pixels[..., 2]
        counts = np.bincount(keys.ravel(), minlength=4096)
        dominant = float(counts.max() / keys.size)
        gy, gx = np.gradient(values)
        edge = float(np.mean(np.hypot(gx, gy)))
        return std, dominant, edge
    except (OSError, ValueError):
        return None


def drop_blank_hits(
    hits: list[tuple[int, float]],
    rows: dict[int, dict],
    uniformity_lookup,
    blank_std: float,
    blank_dominant: float,
    blank_edge: float,
    k: int,
) -> tuple[list[tuple[int, float]], int]:
    """Keep the top-k hits whose frames are not near-uniform.

    Uniform frames (black t=0 frames, white flashes, solid-color wipes) are
    hubs in the SigLIP-2 embedding space: they win nearly every query
    regardless of its text, so they have to be filtered after vector search.
    """
    kept: list[tuple[int, float]] = []
    filtered = 0
    for pk, score in hits:
        row = rows.get(pk)
        if row is None:
            continue
        features = uniformity_lookup(pk, row)
        if features is not None:
            std, dominant, edge = features
            if std <= blank_std or edge <= blank_edge or dominant >= blank_dominant:
                filtered += 1
                continue
        kept.append((pk, score))
        if len(kept) >= k:
            break
    return kept, filtered


def build_results(hits: list[tuple[int, float]], rows: dict[int, dict]) -> list[dict]:
    results = []
    for pk, score in hits:
        row = rows.get(pk)
        if row is None:
            continue
        shot = None
        if row["shot_id"] is not None and row["shot_start_ms"] is not None:
            shot = {
                "shot_id": int(row["shot_id"]),
                "t_start_ms": int(row["shot_start_ms"]),
                "t_end_ms": int(row["shot_end_ms"]),
            }
        results.append(
            {
                "keyframe_pk": int(pk),
                "video_id": row["video_id"],
                "kf_n": int(row["kf_n"]),
                "t_ms": int(row["t_ms"]),
                "t_label": format_timestamp(int(row["t_ms"])),
                "score": float(score),
                "image_url": _media_url(row["image_path"]),
                "thumb_url": _media_url(row["thumb_path"]),
                "shot": shot,
                "video_url": f"/api/video/{row['video_id']}#t={int(row['t_ms']) / 1000.0:.3f}",
            }
        )
    return results


class SearchEngine:
    def __init__(
        self,
        db_path: str | Path,
        faiss_path: str | Path,
        manifest_path: str | Path,
        model_id: str,
        text_template: str = "{}",
        *,
        blank_std: float = 10.0,
        blank_dominant: float = 0.92,
        blank_edge: float = 1.5,
    ):
        self.db_path = Path(db_path)
        self.index_dir = self.db_path.parent
        self.model_id = model_id
        self.text_template = text_template or "{}"
        self.blank_std = blank_std
        self.blank_dominant = blank_dominant
        self.blank_edge = blank_edge
        self._uniformity_cache: dict[int, tuple[float, float, float]] = {}
        self.store = FaissStore.load(
            db_path, faiss_path, manifest_path, expected_model=model_id
        )
        self.model, self.processor = load_model(model_id)

    def frame_uniformity(self, pk: int, row: dict) -> tuple[float, float, float] | None:
        cached = self._uniformity_cache.get(pk)
        if cached is not None:
            return cached
        relative = row.get("thumb_path") or row.get("image_path")
        if not relative:
            return None
        value = frame_uniformity(self.index_dir / relative)
        if value is not None:
            self._uniformity_cache[pk] = value
        return value

    def encode_text(self, query: str) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        text = self.text_template.format(query) if "{}" in self.text_template else query
        inputs = self.processor(text=[text], return_tensors="pt")
        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = functional.normalize(features.float(), p=2, dim=-1)
        return features.cpu().numpy().astype(np.float32)

    def search(self, query: str, k: int = 20) -> dict:
        query = (query or "").strip()
        started = time.perf_counter()
        if not query:
            return {
                "query": query,
                "k": k,
                "encode_ms": 0.0,
                "search_ms": 0.0,
                "latency_ms": 0.0,
                "blank_filtered": 0,
                "results": [],
            }

        encode_started = time.perf_counter()
        vector = self.encode_text(query)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        search_started = time.perf_counter()
        # Over-fetch so blank hub frames can be dropped without returning
        # fewer than k real results.
        fetch = min(self.store.ntotal, max(k * 4, k + 128))
        hits = self.store.search_ids(vector, fetch)
        search_ms = (time.perf_counter() - search_started) * 1000.0

        pks = [pk for pk, _ in hits]
        conn = db.connect(self.db_path)
        try:
            rows = db.get_keyframes_by_pks(conn, pks)
        finally:
            conn.close()

        hits, blank_filtered = drop_blank_hits(
            hits,
            rows,
            self.frame_uniformity,
            self.blank_std,
            self.blank_dominant,
            self.blank_edge,
            k,
        )
        results = build_results(hits, rows)

        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "query": query,
            "k": k,
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "latency_ms": round(total_ms, 2),
            "blank_filtered": blank_filtered,
            "results": results,
        }
