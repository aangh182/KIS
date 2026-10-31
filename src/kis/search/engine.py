"""Text query -> SigLIP-2 text tower -> FAISS -> SQLite metadata."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from kis import db
from kis.index.store import FaissStore
from kis.ingest.embed import load_model

_VI_SPECIFIC = set(
    "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
)
_VI_WORDS = {
    "nguoi", "dang", "trong", "tai", "cua", "va", "mot", "hai", "ba", "nhung",
    "voi", "cho", "tren", "duoi", "truoc", "sau", "khi", "la", "co", "khong",
    "cac", "nhu", "de", "tu", "den", "ra", "vao", "len", "xuong", "cung",
    "rat", "nhieu", "it", "hay", "thi", "ma", "nay", "do", "kieu", "mau",
    "ao", "quan", "xe", "nha", "duong", "bien", "troi", "nguoi", "dan",
    "chuong", "trinh", "truong", "quay", "phat", "bieu", "hoi", "nghi",
}


def looks_vietnamese(text: str) -> bool:
    lowered = text.lower()
    if any(ch in _VI_SPECIFIC for ch in lowered):
        return True
    words = re.findall(r"[a-z]+", lowered)
    return any(word in _VI_WORDS for word in words)


def _load_translator(model_id: str):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).eval()
    return tokenizer, model



def format_timestamp(t_ms: int) -> str:
    total_ms = max(0, int(t_ms))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _media_url(path: str | None) -> str | None:
    if not path:
        return None
    return "/media/" + path.replace("\\", "/").lstrip("/")


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
        translate_query: bool = False,
        translate_model: str = "Helsinki-NLP/opus-mt-vi-en",
    ):
        self.db_path = Path(db_path)
        self.model_id = model_id
        self.text_template = text_template or "{}"
        self.store = FaissStore.load(
            db_path, faiss_path, manifest_path, expected_model=model_id
        )
        self.model, self.processor = load_model(model_id)
        self.translator = _load_translator(translate_model) if translate_query else None

    def translate(self, query: str) -> str:
        import torch

        tokenizer, model = self.translator
        batch = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=64)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

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
                "translated_query": None,
                "k": k,
                "translate_ms": 0.0,
                "encode_ms": 0.0,
                "search_ms": 0.0,
                "latency_ms": 0.0,
                "results": [],
            }

        translated = None
        translate_ms = 0.0
        encoded_query = query
        if self.translator is not None and looks_vietnamese(query):
            translate_started = time.perf_counter()
            translated = self.translate(query)
            translate_ms = (time.perf_counter() - translate_started) * 1000.0
            if translated:
                encoded_query = translated

        encode_started = time.perf_counter()
        vector = self.encode_text(encoded_query)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        search_started = time.perf_counter()
        hits = self.store.search_ids(vector, k)
        search_ms = (time.perf_counter() - search_started) * 1000.0

        pks = [pk for pk, _ in hits]
        conn = db.connect(self.db_path)
        try:
            rows = db.get_keyframes_by_pks(conn, pks)
        finally:
            conn.close()

        results = build_results(hits, rows)

        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "query": query,
            "translated_query": translated,
            "k": k,
            "translate_ms": round(translate_ms, 2),
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "latency_ms": round(total_ms, 2),
            "results": results,
        }
