"""M2: optional illumination correction.

Off by default.  The preferred fix is physical: diffuse light, locked
exposure and a fixed jig.  Software correction is a measured fallback, never
an automatic step, because aggressive equalisation can hide discolouration
that is itself a defect.
"""

from __future__ import annotations

import cv2
import numpy as np


def _background_correction(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    field = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma))
    field = np.maximum(field, 1.0)
    target = float(gray.mean())
    gain = np.clip(target / field, 0.5, 2.0)
    corrected = img_bgr.astype(np.float32) * gain[..., None]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _exposure_normalization(
    img_bgr: np.ndarray,
    target: float = 128.0,
    border_frac: float = 0.05,
) -> np.ndarray:
    """Scale the whole image so the background sits at ``target``.

    The background is estimated from the border, which is assumed to show the
    conveyor/backdrop rather than the part.  A single scalar gain is applied to
    all three channels, so hue and saturation (and therefore discolouration
    defects) are preserved -- only global exposure changes are removed.

    This is the software fallback for a test/train lighting drift.  Prefer
    locking exposure and lighting physically.
    """

    height, width = img_bgr.shape[:2]
    band = max(1, int(round(min(height, width) * float(border_frac))))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    border = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    ).astype(np.float32)
    level = max(float(np.median(border)), 1.0)
    gain = float(np.clip(float(target) / level, 0.5, 2.0))
    corrected = img_bgr.astype(np.float32) * gain
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _gamma_correction(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    inverse = 1.0 / float(gamma)
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** inverse * 255.0).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def _clahe(img_bgr: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid), int(tile_grid)),
    )
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def apply(img_bgr: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Apply the configured illumination correction, or return unchanged."""

    if not cfg or not bool(cfg.get("enabled", False)):
        return img_bgr

    method = str(cfg.get("method", "background"))
    if method == "background":
        return _background_correction(img_bgr, float(cfg.get("sigma", 40.0)))
    if method == "exposure":
        return _exposure_normalization(
            img_bgr,
            float(cfg.get("target", 128.0)),
            float(cfg.get("border_frac", 0.05)),
        )
    if method == "gamma":
        return _gamma_correction(img_bgr, float(cfg.get("gamma", 1.0)))
    if method == "clahe":
        return _clahe(
            img_bgr,
            float(cfg.get("clip_limit", 2.0)),
            int(cfg.get("tile_grid", 8)),
        )
    raise ValueError(
        f"unknown illumination method {method!r} "
        "(expected 'background', 'exposure', 'gamma' or 'clahe')"
    )
