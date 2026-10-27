from __future__ import annotations

import sqlite3
import unittest

from kis import db
from kis.eval.coverage import _covered_ms, compute_coverage, gate_passes


class CoverageTest(unittest.TestCase):
    def test_covered_union(self):
        self.assertEqual(_covered_ms([], 10000, 1000), 0)
        self.assertEqual(_covered_ms([1000, 5000], 10000, 1000), 4000)
        self.assertEqual(_covered_ms([0, 1000, 2000], 3000, 1000), 3000)

    def test_dense_grid_passes_gate(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        db.migrate(conn)
        conn.execute(
            "INSERT INTO videos (video_id, pack_id, video_path, duration_ms) VALUES (?,?,?,?)",
            ("L21_V001", "L21", "/tmp/x.mp4", 10000),
        )
        rows = [
            {
                "video_id": "L21_V001",
                "kf_n": i + 1,
                "frame_idx": i,
                "t_ms": t,
                "pts": t / 1000,
                "shot_id": 0,
                "image_path": "",
                "thumb_path": "",
                "source": "grid",
                "phash": 0,
            }
            for i, t in enumerate(range(0, 10000, 1000))
        ]
        db.replace_keyframes(conn, "L21_V001", rows)
        summary = compute_coverage(conn)
        self.assertAlmostEqual(summary["coverage_1s"], 1.0)
        self.assertEqual(summary["max_gap_p99"], 1000)
        self.assertTrue(gate_passes(summary))
        conn.close()

    def test_empty_index_fails_gate(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        db.migrate(conn)
        conn.execute(
            "INSERT INTO videos (video_id, pack_id, video_path, duration_ms) VALUES (?,?,?,?)",
            ("L21_V001", "L21", "/tmp/x.mp4", 10000),
        )
        summary = compute_coverage(conn)
        self.assertEqual(summary["coverage_1s"], 0.0)
        self.assertEqual(summary["max_gap_p99"], 10000)
        self.assertFalse(gate_passes(summary))
        conn.close()


if __name__ == "__main__":
    unittest.main()
