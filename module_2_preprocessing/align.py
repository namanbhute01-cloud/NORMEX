"""M2: optional pose alignment to a reference good image.

A fixed jig is the first line of defence.  Software alignment is off by
default because warping a part can hide a real deformation.  When enabled it
uses ECC (translation + rotation only, no homography) and refuses corrections
larger than the configured limits.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def load_reference(path: str | Path) -> np.ndarray:
    """Read a reference good image as BGR."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read alignment reference image: {path}")
    return image


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0


def limits(cfg: dict | None) -> tuple[float, float]:
    cfg = cfg or {}
    return (
        float(cfg.get("max_shift_px", 8.0)),
        float(cfg.get("max_rotation_deg", 2.0)),
    )


def estimate(reference_bgr: np.ndarray, image_bgr: np.ndarray, cfg: dict | None) -> dict:
    """Estimate an ECC euclidean warp from reference to image coordinates.

    OpenCV's ECC warp is consumed with ``WARP_INVERSE_MAP`` by :func:`apply`,
    which aligns ``image_bgr`` to ``reference_bgr``.  Returns a dict with
    ``ok``, ``applied``, ``cc``, ``shift_px``, ``rotation_deg``, ``warp`` and
    ``message``.  A failed or too-large correction is not applied; the caller
    should treat it as a bad image.
    """

    reference = _gray_float(reference_bgr)
    image = _gray_float(image_bgr)
    if reference.shape != image.shape:
        return {
            "ok": False,
            "applied": False,
            "cc": 0.0,
            "shift_px": float("nan"),
            "rotation_deg": float("nan"),
            "warp": None,
            "message": (
                f"alignment reference is {reference.shape[::-1]}, "
                f"image is {image.shape[::-1]}"
            ),
        }

    max_shift, max_rotation = limits(cfg)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    try:
        cc, warp = cv2.findTransformECC(
            reference, image, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error as exc:
        return {
            "ok": False,
            "applied": False,
            "cc": 0.0,
            "shift_px": float("nan"),
            "rotation_deg": float("nan"),
            "warp": None,
            "message": f"alignment failed: {exc}",
        }

    shift_px = float(math.hypot(float(warp[0, 2]), float(warp[1, 2])))
    rotation_deg = float(math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0]))))
    if shift_px > max_shift or abs(rotation_deg) > max_rotation:
        return {
            "ok": False,
            "applied": False,
            "cc": float(cc),
            "shift_px": shift_px,
            "rotation_deg": rotation_deg,
            "warp": None,
            "message": (
                f"alignment correction too large "
                f"(shift={shift_px:.1f}px, rotation={rotation_deg:.1f}deg)"
            ),
        }

    return {
        "ok": True,
        "applied": True,
        "cc": float(cc),
        "shift_px": shift_px,
        "rotation_deg": rotation_deg,
        "warp": warp,
        "message": "",
    }


def apply(image_bgr: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Warp ``image_bgr`` into the reference frame."""

    height, width = image_bgr.shape[:2]
    return cv2.warpAffine(
        image_bgr,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def align(
    image_bgr: np.ndarray, reference_bgr: np.ndarray, cfg: dict | None
) -> tuple[np.ndarray, dict]:
    """Align one image, returning ``(image, info)``; failure leaves it as-is."""

    info = estimate(reference_bgr, image_bgr, cfg)
    if info["ok"] and info["warp"] is not None:
        return apply(image_bgr, info["warp"]), info
    return image_bgr, info
