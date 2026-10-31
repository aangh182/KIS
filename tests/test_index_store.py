from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kis import db
from kis.index.build import build_index
from kis.index.store import FaissStore
from kis.ingest.embed import shard_paths, write_shard

MODEL = "test-model"


def normalized(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(rows, dim)).astype(np.float32)
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array.astype(np.float16)


def seed_db(path: Path, video_id: str, count: int) -> list[int]:
    conn = db.connect(path)
    db.migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO videos (video_id, pack_id, video_path, duration_ms)"
            " VALUES (?, ?, ?, ?)",
            (video_id, video_id.split("_", 1)[0], f"/videos/{video_id}.mp4", count * 1000),
        )
        conn.executemany(
            "INSERT INTO keyframes (video_id, kf_n, t_ms, image_path, thumb_path)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (video_id, n, n * 1000, f"keyframes/L13/{video_id}/{n}.jpg", f"thumbnails/L13/{video_id}/{n}.jpg")
                for n in range(1, count + 1)
            ],
        )
    pks = [row["keyframe_pk"] for row in db.get_keyframes_for_video(conn, video_id)]
    conn.close()
    return pks


def write_video_shard(
    embeddings: Path, video_id: str, pks: list[int], dim: int, seed: int = 0
) -> np.ndarray:
    array = normalized(len(pks), dim, seed=seed)
    npy, meta = shard_paths(embeddings, video_id.split("_", 1)[0], video_id)
    write_shard(
        npy,
        meta,
        array,
        video_id=video_id,
        model_id=MODEL,
        keyframe_pks=pks,
        dtype="float16",
    )
    return array


class BuildAndStoreTest(unittest.TestCase):
    def test_build_and_search_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "meta.db"
            embeddings = root / "embeddings"
            faiss_path = root / "faiss.index"
            manifest_path = root / "manifest.json"

            pks_a = seed_db(db_path, "L13_V001", 3)
            pks_b = seed_db(db_path, "L13_V002", 2)
            array_a = write_video_shard(embeddings, "L13_V001", pks_a, 8, seed=1)
            write_video_shard(embeddings, "L13_V002", pks_b, 8, seed=2)

            result = build_index(
                db_path,
                embeddings,
                faiss_path,
                manifest_path,
                model_id=MODEL,
            )
            self.assertEqual(result["ntotal"], 5)
            self.assertEqual(result["dim"], 8)
            self.assertEqual(result["videos"], 2)
            self.assertFalse(result["partial"])

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["embed"]["model"], MODEL)
            self.assertEqual(manifest["counts"]["keyframes_db"], 5)

            store = FaissStore.load(db_path, faiss_path, manifest_path, expected_model=MODEL)
            hits = store.search_ids(array_a[1].astype(np.float32), 3)
            self.assertEqual(hits[0][0], pks_a[1])

    def test_build_rejects_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "meta.db"
            embeddings = root / "embeddings"
            pks_a = seed_db(db_path, "L13_V001", 2)
            pks_b = seed_db(db_path, "L13_V002", 2)
            write_video_shard(embeddings, "L13_V001", pks_a, 8)
            npy, meta = shard_paths(embeddings, "L13", "L13_V002")
            write_video_shard(embeddings, "L13_V002", pks_b, 8)
            npy.unlink()
            meta.unlink()

            with self.assertRaises(RuntimeError):
                build_index(
                    db_path,
                    embeddings,
                    root / "faiss.index",
                    root / "manifest.json",
                    model_id=MODEL,
                )

    def test_partial_build_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "meta.db"
            embeddings = root / "embeddings"
            pks_a = seed_db(db_path, "L13_V001", 2)
            seed_db(db_path, "L13_V002", 2)
            write_video_shard(embeddings, "L13_V001", pks_a, 8)

            result = build_index(
                db_path,
                embeddings,
                root / "faiss.index",
                root / "manifest.json",
                model_id=MODEL,
                allow_partial=True,
            )
            self.assertTrue(result["partial"])
            self.assertEqual(result["ntotal"], 2)

    def test_store_detects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "meta.db"
            embeddings = root / "embeddings"
            pks_a = seed_db(db_path, "L13_V001", 2)
            write_video_shard(embeddings, "L13_V001", pks_a, 8)
            build_index(
                db_path,
                embeddings,
                root / "faiss.index",
                root / "manifest.json",
                model_id=MODEL,
            )

            conn = db.connect(db_path)
            with conn:
                conn.execute(
                    "INSERT INTO keyframes (video_id, kf_n, t_ms)"
                    " VALUES ('L13_V001', 99, 99000)"
                )
            conn.close()

            with self.assertRaises(RuntimeError):
                FaissStore.load(db_path, root / "faiss.index", root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
