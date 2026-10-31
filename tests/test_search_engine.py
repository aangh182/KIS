from __future__ import annotations

import unittest

from kis.search.engine import build_results, format_timestamp, looks_vietnamese


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


if __name__ == "__main__":
    unittest.main()
