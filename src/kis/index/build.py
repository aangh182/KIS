"""Build the FAISS index from per-video embedding shards."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kis import db
from kis.ingest.embed import load_sidecar, sha256_file


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _collect_shards(embeddings_dir: Path, model_id: str) -> tuple[list[np.ndarray], list[int], list[dict]]:
    meta_files = sorted(embeddings_dir.glob("*/*.json"))
    if not meta_files:
        raise RuntimeError(f"no embedding shards under {embeddings_dir}; run `kis embed` first")

    arrays: list[np.ndarray] = []
    ids: list[int] = []
    shards: list[dict] = []
    dim: int | None = None

    for meta_path in meta_files:
        meta = load_sidecar(meta_path)
        if meta is None:
            raise RuntimeError(f"invalid sidecar: {meta_path}")
        if meta.get("model") != model_id:
            raise RuntimeError(
                f"{meta_path}: model {meta.get('model')!r} != expected {model_id!r}"
            )

        npy_path = meta_path.with_suffix(".npy")
        if not npy_path.exists():
            raise RuntimeError(f"missing shard file: {npy_path}")
        if meta.get("sha256") != sha256_file(npy_path):
            raise RuntimeError(f"checksum mismatch: {npy_path}")

        array = np.load(npy_path).astype(np.float32)
        if array.ndim != 2:
            raise RuntimeError(f"{npy_path}: expected 2-D array, got shape {array.shape}")
        if dim is None:
            dim = int(array.shape[1])
        elif int(array.shape[1]) != dim:
            raise RuntimeError(f"{npy_path}: dim {array.shape[1]} != {dim}")

        keyframe_pks = [int(pk) for pk in meta.get("keyframe_pks", [])]
        if array.shape[0] != len(keyframe_pks):
            raise RuntimeError(
                f"{npy_path}: {array.shape[0]} vectors != {len(keyframe_pks)} keyframe ids"
            )
        norms = np.linalg.norm(array, axis=1)
        if array.shape[0] and not np.all((norms > 0.98) & (norms < 1.02)):
            raise RuntimeError(f"{npy_path}: vectors are not L2-normalized")

        arrays.append(array)
        ids.extend(keyframe_pks)
        shards.append(
            {
                "video_id": meta.get("video_id"),
                "count": int(array.shape[0]),
                "sha256": meta.get("sha256"),
            }
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate keyframe ids across shards under {embeddings_dir}")
    return arrays, ids, shards


def build_index(
    db_path: str | Path,
    embeddings_dir: str | Path,
    faiss_path: str | Path,
    manifest_path: str | Path,
    *,
    model_id: str,
    sampling: dict | None = None,
    allow_partial: bool = False,
) -> dict:
    embeddings_dir = Path(embeddings_dir)
    faiss_path = Path(faiss_path)
    manifest_path = Path(manifest_path)

    import faiss

    conn = db.connect(db_path)
    db.migrate(conn)
    try:
        db_pks = set(db.get_all_keyframe_pks(conn))
        nkeyframes_db = len(db_pks)
        if nkeyframes_db == 0:
            raise RuntimeError("no keyframes in database; run `kis keyframes` first")

        arrays, ids, shards = _collect_shards(embeddings_dir, model_id)
        ids_set = set(ids)
        missing = db_pks - ids_set
        excess = ids_set - db_pks
        if not allow_partial and (missing or excess):
            raise RuntimeError(
                "embedding shards do not match keyframes table: "
                f"{len(missing)} missing, {len(excess)} unknown. "
                "Run `kis embed` (with --force if keyframes changed)."
            )
        if allow_partial and excess:
            raise RuntimeError(
                f"{len(excess)} embedded keyframe ids no longer exist in the database; "
                "re-run `kis embed --force`"
            )
        if not allow_partial and len(ids) != nkeyframes_db:
            raise RuntimeError(f"only {len(ids)}/{nkeyframes_db} keyframes embedded")

        matrix = np.concatenate(arrays, axis=0)
        dim = int(matrix.shape[1])
        vector_ids = np.asarray(ids, dtype=np.int64)

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        index.add_with_ids(matrix, vector_ids)
        if int(index.ntotal) != len(ids):
            raise RuntimeError(f"FAISS ntotal {index.ntotal} != {len(ids)} vectors")

        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = faiss_path.with_suffix(".index.tmp")
        faiss.write_index(index, str(tmp_index))
        os.replace(tmp_index, faiss_path)

        from importlib.metadata import version

        # Fold the WAL back into meta.db so the recorded checksum is stable.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "partial": bool(allow_partial),
            "embed": {
                "model": model_id,
                "dim": dim,
                "dtype": "float16",
                "normalized": True,
                "transformers": version("transformers"),
            },
            "index": {
                "type": "IndexIDMap2(IndexFlatIP)",
                "ntotal": int(index.ntotal),
                "faiss_sha256": sha256_file(faiss_path),
            },
            "counts": {
                "videos_indexed": len(shards),
                "keyframes_indexed": len(ids),
                "keyframes_db": nkeyframes_db,
            },
            "sampling": sampling or {},
            "checksums": {"meta_db_sha256": sha256_file(Path(db_path))},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_manifest, manifest_path)

        return {
            "ntotal": int(index.ntotal),
            "dim": dim,
            "videos": len(shards),
            "keyframes_db": nkeyframes_db,
            "partial": bool(allow_partial),
            "faiss_bytes": faiss_path.stat().st_size,
        }
    finally:
        conn.close()
