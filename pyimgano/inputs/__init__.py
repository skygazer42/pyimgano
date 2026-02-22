"""Input utilities for industrial inference.

The core idea is to avoid "auto guessing" image formats in production.
Callers must declare the input format explicitly and `pyimgano` converts to a
single canonical numpy representation:

- RGB
- uint8
- HWC
"""

from __future__ import annotations

from .image_format import ImageFormat, normalize_numpy_image, parse_image_format

__all__ = [
    "ImageFormat",
    "normalize_numpy_image",
    "parse_image_format",
]
