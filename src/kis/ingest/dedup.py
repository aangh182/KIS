"""Perceptual hash helpers for near-duplicate keyframe suppression."""

from __future__ import annotations

_U64_MASK = (1 << 64) - 1
_I63 = 1 << 63


def phash_to_int(value) -> int:
    """64-bit hash as a signed int so it fits SQLite's INTEGER type."""
    unsigned = int(str(value), 16) & _U64_MASK
    return unsigned - (1 << 64) if unsigned >= _I63 else unsigned


def hamming(a: int, b: int) -> int:
    return ((a & _U64_MASK) ^ (b & _U64_MASK)).bit_count()


def is_near_duplicate(a: int, b: int, threshold: int = 4) -> bool:
    return hamming(a, b) <= threshold
