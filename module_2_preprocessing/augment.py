"""M2: defect-safe augmentation for training only.

Only geometric/lighting variation that a *good* part can legitimately show
is applied.  Blur, noise, cutout or cut-paste are forbidden: they mimic
defects and would teach the model that defects are normal.
"""

from __future__ import annotations

import cv2
import numpy as np


def augment(img_bgr: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Apply mild shift / rotation / brightness / contrast (and optional flip).

    Deterministic for a given ``rng`` state.
    """

    h, w = img_bgr.shape[:2]
    shift = float(cfg.get("shift", 0.0))
    rot_deg = float(cfg.get("rot_deg", 0.0))

    dx = float(rng.uniform(-shift, shift)) * w if shift > 0 else 0.0
    dy = float(rng.uniform(-shift, shift)) * h if shift > 0 else 0.0
    angle = float(rng.uniform(-rot_deg, rot_deg)) if rot_deg > 0 else 0.0

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    out = cv2.warpAffine(
        img_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if bool(cfg.get("flip", False)) and rng.random() < 0.5:
        out = cv2.flip(out, 1)

    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    if brightness > 0 or contrast > 0:
        alpha = 1.0 + float(rng.uniform(-contrast, contrast)) if contrast > 0 else 1.0
        beta = float(rng.uniform(-brightness, brightness)) * 255.0 if brightness > 0 else 0.0
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return out
