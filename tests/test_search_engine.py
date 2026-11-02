from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kis.search.engine import (
    build_results,
    drop_blank_hits,
    format_timestamp,
    frame_uniformity,
    looks_vietnamese,
)


class LanguageTest(unittest.TestCase):
    def test_detects_vietnamese(self):
        self.assertTrue(looks_vietnamese("một người dẫn chương trình"))
        self.assertTrue(looks_vietnamese("nguoi dan chuong trinh"))
        self.assertFalse(looks_vietnamese("a woman in a white shirt"))
        self.assertFalse(looks_vietnamese("red and white logo"))


class FormatTest(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00.000")
        self.assertEqual(format_timestamp(61234), "01:01.234")
        self.assertEqual(format_timestamp(3_600_001), "60:00.001")
        self.assertEqual(format_timestamp(-5), "00:00.000")


class ResultTest(unittest.TestCase):
    def test_build_results(self):
        rows = {
            7: {
                "keyframe_pk": 7,
                "video_id": "L13_V001",
                "kf_n": 3,
                "t_ms": 61234,
                "shot_id": 2,
                "image_path": "keyframes/L13/L13_V001/3.jpg",
                "thumb_path": "thumbnails/L13/L13_V001/3.jpg",
                "shot_start_ms": 60000,
                "shot_end_ms": 63500,
            }
        }
        results = build_results([(7, 0.4321), (99, 0.1)], rows)
        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item["keyframe_pk"], 7)
        self.assertEqual(item["t_label"], "01:01.234")
        self.assertEqual(item["shot"], {"shot_id": 2, "t_start_ms": 60000, "t_end_ms": 63500})
        self.assertEqual(item["image_url"], "/media/keyframes/L13/L13_V001/3.jpg")
        self.assertEqual(item["video_url"], "/api/video/L13_V001#t=61.234")

    def test_windows_paths_and_missing_shot(self):
        rows = {
            1: {
                "keyframe_pk": 1,
                "video_id": "L13_V001",
                "kf_n": 1,
                "t_ms": 1000,
                "shot_id": None,
                "image_path": "keyframes\\L13\\L13_V001\\1.jpg",
                "thumb_path": "thumbnails\\L13\\L13_V001\\1.jpg",
                "shot_start_ms": None,
                "shot_end_ms": None,
            }
        }
        item = build_results([(1, 0.2)], rows)[0]
        self.assertIsNone(item["shot"])
        self.assertEqual(item["image_url"], "/media/keyframes/L13/L13_V001/1.jpg")


class BlankFilterTest(unittest.TestCase):
    def setUp(self):
        self.rows = {
            pk: {"keyframe_pk": pk, "thumb_path": f"thumbnails/{pk}.jpg"}
            for pk in range(1, 12)
        }
        self.features = {
            1: (0.0, 1.000, 0.00),
            2: (5.0, 0.950, 0.10),
            3: (40.0, 0.960, 1.24),
            4: (60.0, 0.050, 17.0),
            5: (14.0, 0.100, 6.0),
            6: (50.0, 0.300, 13.0),
            7: (19.5, 0.830, 4.25),
            8: (15.1, 0.895, 2.22),
            9: (50.7, 0.875, 1.25),
            10: (28.0, 0.130, 1.01),
        }

    def lookup(self, pk, row):
        return self.features.get(pk)

    def test_drops_uniform_and_solid_color_frames(self):
        hits = [(pk, 1.0 - pk / 100) for pk in range(1, 11)]
        kept, filtered = drop_blank_hits(
            hits, self.rows, self.lookup, 10.0, 0.92, 1.5, 3
        )
        self.assertEqual([pk for pk, _ in kept], [4, 5, 6])
        self.assertEqual(filtered, 3)

    def test_keeps_dark_scenes_and_logo_screens(self):
        hits = [(7, 0.9), (8, 0.8), (9, 0.7)]
        kept, filtered = drop_blank_hits(
            hits, self.rows, self.lookup, 10.0, 0.92, 1.5, 5
        )
        self.assertEqual([pk for pk, _ in kept], [7, 8])
        self.assertEqual(filtered, 1)

    def test_drops_smooth_gradient_frames(self):
        kept, filtered = drop_blank_hits(
            [(10, 0.9)], self.rows, self.lookup, 10.0, 0.92, 1.5, 5
        )
        self.assertEqual(kept, [])
        self.assertEqual(filtered, 1)

    def test_unknown_features_are_kept(self):
        hits = [(11, 0.9), (12, 0.8)]
        kept, filtered = drop_blank_hits(
            hits, self.rows, self.lookup, 10.0, 0.92, 1.5, 2
        )
        self.assertEqual([pk for pk, _ in kept], [11])
        self.assertEqual(filtered, 0)

    def test_missing_row_is_skipped(self):
        hits = [(99, 0.9), (5, 0.8)]
        kept, _ = drop_blank_hits(hits, self.rows, self.lookup, 10.0, 0.92, 1.5, 5)
        self.assertEqual([pk for pk, _ in kept], [5])

    def test_frame_uniformity_separates_empty_from_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            blank = Path(tmp) / "blank.jpg"
            Image.new("RGB", (64, 64), (255, 255, 255)).save(blank)
            std, dominant, _ = frame_uniformity(blank)
            self.assertEqual(std, 0.0)
            self.assertGreater(dominant, 0.9)

            wipe = Path(tmp) / "wipe.jpg"
            image = Image.new("RGB", (64, 64), (200, 20, 20))
            image.paste((255, 255, 255), (0, 0, 64, 2))
            image.save(wipe)
            std, dominant, _ = frame_uniformity(wipe)
            self.assertGreater(std, 10.0)
            self.assertGreater(dominant, 0.85)

            content = Path(tmp) / "content.jpg"
            image = Image.new("RGB", (64, 64), (0, 0, 0))
            image.paste((255, 255, 255), (0, 0, 32, 64))
            image.paste((0, 0, 255), (32, 0, 64, 32))
            image.save(content)
            std, dominant, _ = frame_uniformity(content)
            self.assertGreater(std, 10.0)
            self.assertLess(dominant, 0.85)

            self.assertIsNone(frame_uniformity(Path(tmp) / "missing.jpg"))


if __name__ == "__main__":
    unittest.main()
