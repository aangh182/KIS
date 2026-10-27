from __future__ import annotations

import unittest

from kis.ingest.dedup import hamming, is_near_duplicate, phash_to_int


class DedupTest(unittest.TestCase):
    def test_signed_storage_range(self):
        for hex_value in ["0000000000000000", "7fffffffffffffff", "8000000000000000", "ffffffffffffffff"]:
            value = phash_to_int(hex_value)
            self.assertGreaterEqual(value, -(1 << 63))
            self.assertLess(value, 1 << 63)

    def test_hamming_uses_unsigned_64_bits(self):
        zero = phash_to_int("0" * 16)
        ones = phash_to_int("f" * 16)
        self.assertEqual(hamming(zero, zero), 0)
        self.assertEqual(hamming(zero, ones), 64)
        self.assertFalse(is_near_duplicate(zero, ones))
        self.assertTrue(is_near_duplicate(zero, 1))


if __name__ == "__main__":
    unittest.main()
