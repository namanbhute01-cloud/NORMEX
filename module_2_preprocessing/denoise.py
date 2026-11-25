"""M2: optional, conservative denoising.

Off by default.  A scratch or fine crack is high-frequency signal, so blur
can erase the defect the system is meant to find.  Enable only after measuring
real camera noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply(img_bgr: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Apply the configured denoiser, or return the image unchanged."""

    if not cfg or not bool(cfg.get("enabled", False)):
        return img_bgr

    method = str(cfg.get("method", "median"))
    if method == "median":
        ksize = int(cfg.get("ksize", 3))
        if ksize < 3 or ksize % 2 == 0:
            raise ValueError(f"median denoise ksize must be odd and >= 3, got {ksize}")
        return cv2.medianBlur(img_bgr, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(
            img_bgr,
            d=int(cfg.get("d", 5)),
            sigmaColor=float(cfg.get("sigma_color", 25.0)),
            sigmaSpace=float(cfg.get("sigma_space", 25.0)),
        )
    raise ValueError(f"unknown denoise method {method!r} (expected 'median' or 'bilateral')")
