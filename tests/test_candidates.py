from __future__ import annotations

import unittest

from kis.ingest.candidates import (
    grid_timestamps,
    profile_key,
    select_candidates,
)

CFG = {
    "grid_s": 3.0,
    "max_per_shot": 6,
    "min_shot_s": 0.4,
    "long_shot_s": 10.0,
    "min_gap_ms": 800,
}


def shots(*ranges):
    return [
        {"shot_id": i, "t_start_ms": a, "t_end_ms": b, "start_frame": 0}
        for i, (a, b) in enumerate(ranges)
    ]


class ProfileTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(profile_key("hybrid"), "v3")
        self.assertEqual(profile_key("iframe"), "v0")
        self.assertEqual(profile_key("shot"), "v2")
        with self.assertRaises(ValueError):
            profile_key("nope")

    def test_grid_timestamps(self):
        self.assertEqual(grid_timestamps(3500, 1000), [0, 1000, 2000, 3000])


class SelectionTest(unittest.TestCase):
    def test_v0_uses_iframes(self):
        picks, degraded = select_candidates(
            shots((0, 5000)), [0, 2000, 9000], 5000, "v0", CFG
        )
        self.assertFalse(degraded)
        self.assertEqual([p["target_ms"] for p in picks], [0, 2000])

    def test_v1_uses_one_fps_grid(self):
        picks, degraded = select_candidates(shots((0, 5000)), [0], 2500, "v1", CFG)
        self.assertFalse(degraded)
        self.assertEqual([p["target_ms"] for p in picks], [0, 1000, 2000])

    def test_v2_uses_shot_midpoints(self):
        picks, _ = select_candidates(shots((0, 2000), (2000, 6000)), [], 6000, "v2", CFG)
        self.assertEqual([p["target_ms"] for p in picks], [1000, 4000])

    def test_hybrid_uses_iframes_and_grid(self):
        picks, degraded = select_candidates(
            shots((0, 8000), (8000, 12000)),
            [200, 8300],
            12000,
            "hybrid",
            CFG,
        )
        self.assertFalse(degraded)
        times = [p["target_ms"] for p in picks]
        self.assertIn(0, times)
        self.assertIn(3000, times)
        self.assertIn(6000, times)
        self.assertIn(8300, times)

    def test_hybrid_falls_back_to_grid_when_single_shot(self):
        picks, degraded = select_candidates(shots((0, 20000)), [0], 20000, "hybrid", CFG)
        self.assertTrue(degraded)
        self.assertEqual(picks[1]["target_ms"], 1000)
        self.assertTrue(all(p["source"] == "grid" for p in picks))

    def test_long_shot_uses_grid_without_degrading(self):
        picks, degraded = select_candidates(
            shots((0, 2000), (2000, 15000)), [0, 3000], 15000, "hybrid", CFG
        )
        self.assertFalse(degraded)
        self.assertIn(5000, [p["target_ms"] for p in picks])

    def test_long_shot_degrade_is_opt_in(self):
        picks, degraded = select_candidates(
            shots((0, 2000), (2000, 15000)),
            [0, 3000],
            15000,
            "hybrid",
            {**CFG, "degrade_on_long_shot": True},
        )
        self.assertTrue(degraded)
        self.assertEqual(len(picks), 15)

    def test_dedup_min_gap(self):
        picks, _ = select_candidates(
            shots((0, 5000)), [0, 500, 2000], 5000, "v0", CFG
        )
        self.assertEqual([p["target_ms"] for p in picks], [0, 2000])


if __name__ == "__main__":
    unittest.main()
