from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kis.ingest.embed import (
    shard_is_valid,
    shard_paths,
    sha256_file,
    write_shard,
)


def normalized(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(rows, dim)).astype(np.float32)
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array.astype(np.float16)


class ShardTest(unittest.TestCase):
    def test_paths(self):
        npy, meta = shard_paths(Path("index/embeddings"), "L13", "L13_V001")
        self.assertEqual(npy.name, "L13_V001.npy")
        self.assertEqual(meta.name, "L13_V001.json")
        self.assertEqual(npy.parent.name, "L13")

    def test_write_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy, meta = shard_paths(root, "L13", "L13_V001")
            array = normalized(3, 8)
            pks = [11, 12, 13]
            info = write_shard(
                npy,
                meta,
                array,
                video_id="L13_V001",
                model_id="test-model",
                keyframe_pks=pks,
                dtype="float16",
            )
            self.assertEqual(info["count"], 3)
            self.assertEqual(info["dim"], 8)
            self.assertEqual(info["sha256"], sha256_file(npy))
            self.assertTrue(shard_is_valid(npy, meta, "test-model", pks))

    def test_rejects_model_and_id_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy, meta = shard_paths(root, "L13", "L13_V001")
            write_shard(
                npy,
                meta,
                normalized(2, 8),
                video_id="L13_V001",
                model_id="test-model",
                keyframe_pks=[1, 2],
                dtype="float16",
            )
            self.assertFalse(shard_is_valid(npy, meta, "other-model", [1, 2]))
            self.assertFalse(shard_is_valid(npy, meta, "test-model", [1, 3]))

    def test_rejects_tampered_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy, meta = shard_paths(root, "L13", "L13_V001")
            write_shard(
                npy,
                meta,
                normalized(2, 8),
                video_id="L13_V001",
                model_id="test-model",
                keyframe_pks=[1, 2],
                dtype="float16",
            )
            np.save(npy, normalized(2, 8, seed=99))
            self.assertFalse(shard_is_valid(npy, meta, "test-model", [1, 2]))


if __name__ == "__main__":
    unittest.main()
