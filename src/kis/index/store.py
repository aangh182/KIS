"""Load and validate the FAISS index against SQLite and its manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kis import db


class FaissStore:
    def __init__(self, index, manifest: dict, dim: int):
        self.index = index
        self.manifest = manifest
        self.dim = dim

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def partial(self) -> bool:
        return bool(self.manifest.get("partial", False))

    @classmethod
    def load(
        cls,
        db_path: str | Path,
        faiss_path: str | Path,
        manifest_path: str | Path,
        *,
        expected_model: str | None = None,
    ) -> "FaissStore":
        import faiss

        faiss_path = Path(faiss_path)
        manifest_path = Path(manifest_path)
        if not faiss_path.exists() or not manifest_path.exists():
            raise RuntimeError(
                f"missing index artifacts ({faiss_path}, {manifest_path}); run `kis index build`"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(faiss_path))
        dim = int(manifest["embed"]["dim"])

        if int(index.ntotal) != int(manifest["index"]["ntotal"]):
            raise RuntimeError(
                f"FAISS ntotal {index.ntotal} != manifest ntotal {manifest['index']['ntotal']}"
            )
        if int(index.d) != dim:
            raise RuntimeError(f"FAISS dim {index.d} != manifest dim {dim}")
        model = manifest["embed"]["model"]
        if expected_model and model != expected_model:
            raise RuntimeError(
                f"index built with {model!r} but expected {expected_model!r}; rebuild the index"
            )

        conn = db.connect(db_path)
        try:
            nkeyframes_db = db.count_rows(conn, "keyframes")
            if not manifest.get("partial", False) and int(index.ntotal) != nkeyframes_db:
                raise RuntimeError(
                    f"FAISS ntotal {index.ntotal} != keyframes table {nkeyframes_db}; "
                    "run `kis embed` then `kis index build`"
                )
        finally:
            conn.close()

        return cls(index, manifest, dim)

    def search_ids(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector[None, :]
        if vector.shape != (1, self.dim):
            raise RuntimeError(
                f"query shape {vector.shape} != expected (1, {self.dim})"
            )
        k = max(1, min(int(k), self.ntotal))
        scores, ids = self.index.search(np.ascontiguousarray(vector), k)
        return [(int(pk), float(score)) for pk, score in zip(ids[0], scores[0]) if pk >= 0]
