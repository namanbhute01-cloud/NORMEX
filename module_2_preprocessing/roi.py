"""M2: fixed-box region-of-interest extraction.

The first version uses a manually configured box per product (the physical
jig fixes the part).  Automatic segmentation is deliberately deferred: a
second model would add failure modes and is not needed for a fixed camera.
"""

from __future__ import annotations

import numpy as np


def resolve_box(cfg: dict | None, width: int, height: int) -> list[int]:
    """Return ``[x, y, w, h]`` in source pixels.

    A missing/disabled ROI, or zero width/height, means the full frame.
    """

    if not cfg or not bool(cfg.get("enabled", False)):
        return [0, 0, int(width), int(height)]

    x = int(cfg.get("x", 0))
    y = int(cfg.get("y", 0))
    w = int(cfg.get("width", 0))
    h = int(cfg.get("height", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"enabled ROI needs positive width/height, got {w}x{h}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(
            f"ROI [{x}, {y}, {w}, {h}] does not fit inside a {width}x{height} image"
        )
    return [x, y, w, h]


def crop(img: np.ndarray, box: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``img`` to ``[x, y, w, h]`` and return a contiguous copy."""

    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid ROI size {w}x{h}")
    height, width = img.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {box} does not fit inside a {width}x{height} image")
    return np.ascontiguousarray(img[y : y + h, x : x + w])
