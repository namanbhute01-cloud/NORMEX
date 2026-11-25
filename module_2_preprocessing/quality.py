"""M2: image-quality gate.

Blur, exposure, saturation and foreground-occupancy limits are *learned* from
the good images of each product and stored in its profile.  An image outside
those limits is reported as ``BAD_IMAGE`` instead of being guessed at.

Quality is measured on the ROI when an ROI is configured, so the background
does not hide a blurred or badly framed part.
"""

from __future__ import annotations

import cv2
import numpy as np


def validate(img_bgr: np.ndarray) -> list[str]:
    """Return structural problems with an input image (empty list = valid)."""

    problems: list[str] = []
    if not isinstance(img_bgr, np.ndarray):
        return [f"expected a NumPy array, got {type(img_bgr).__name__}"]
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        problems.append(f"expected HxWx3 BGR image, got shape {img_bgr.shape}")
        return problems
    if img_bgr.dtype != np.uint8:
        problems.append(f"expected uint8 pixels, got {img_bgr.dtype}")
    height, width = img_bgr.shape[:2]
    if height < 32 or width < 32:
        problems.append(f"image too small: {width}x{height}")
    return problems


def _foreground_fraction(gray: np.ndarray) -> float:
    """Fraction of pixels that differ from the border/background intensity.

    This is deliberately simple and product-agnostic: estimate the background
    from the image border, then count pixels far from it.  It catches an empty
    frame or a part that has moved out of view; it is not a segmentation model.
    """

    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    ).astype(np.float32)
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    threshold = max(10.0, 4.0 * mad)
    difference = np.abs(gray.astype(np.float32) - background)
    return float((difference > threshold).mean())


def measure(img_bgr: np.ndarray) -> dict:
    """Return blur / brightness / saturation / occupancy statistics."""

    problems = validate(img_bgr)
    if problems:
        raise ValueError("; ".join(problems))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean": float(gray.mean()),
        "sat_frac": float((gray >= 250).mean()),
        "fg_frac": _foreground_fraction(gray),
        "width": int(width),
        "height": int(height),
        "channels": int(img_bgr.shape[2]),
    }


def learn_limits(
    measures: list[dict],
    *,
    check_foreground: bool = True,
    require_same_size: bool = True,
) -> dict:
    """Derive generous limits from good images only."""

    if not measures:
        raise ValueError("learn_limits needs at least one measurement")

    blur = np.array([m["blur"] for m in measures], dtype=np.float64)
    mean = np.array([m["mean"] for m in measures], dtype=np.float64)
    sat = np.array([m["sat_frac"] for m in measures], dtype=np.float64)
    fg = np.array([m.get("fg_frac", 0.0) for m in measures], dtype=np.float64)

    limits = {
        "blur_min": float(0.3 * np.percentile(blur, 5)),
        "mean_min": float(0.6 * mean.min()),
        "mean_max": float(min(255.0, 1.4 * mean.max())),
        "sat_max": float(max(0.05, 2.0 * sat.max())),
        "width": 0,
        "height": 0,
        "fg_min": None,
        "fg_max": None,
    }

    if require_same_size:
        sizes = {(int(m["width"]), int(m["height"])) for m in measures}
        if len(sizes) == 1:
            limits["width"], limits["height"] = sizes.pop()

    if check_foreground:
        limits["fg_min"] = float(max(0.005, 0.5 * np.percentile(fg, 5)))
        limits["fg_max"] = float(min(1.0, 1.5 * np.percentile(fg, 95) + 0.05))

    return limits


def check(measurement: dict, limits: dict) -> tuple[bool, str]:
    """Check one measurement against learned limits."""

    problems: list[str] = []
    if measurement["blur"] < limits["blur_min"]:
        problems.append(
            f"too blurry (blur={measurement['blur']:.1f} < {limits['blur_min']:.1f})"
        )
    if measurement["mean"] < limits["mean_min"]:
        problems.append(
            f"too dark (mean={measurement['mean']:.1f} < {limits['mean_min']:.1f})"
        )
    if measurement["mean"] > limits["mean_max"]:
        problems.append(
            f"too bright (mean={measurement['mean']:.1f} > {limits['mean_max']:.1f})"
        )
    if measurement["sat_frac"] > limits["sat_max"]:
        problems.append(
            f"over-exposed (sat_frac={measurement['sat_frac']:.3f} > {limits['sat_max']:.3f})"
        )

    expected_w = int(limits.get("width", 0) or 0)
    expected_h = int(limits.get("height", 0) or 0)
    if expected_w and expected_h:
        got_w = int(measurement.get("width", expected_w))
        got_h = int(measurement.get("height", expected_h))
        if (got_w, got_h) != (expected_w, expected_h):
            problems.append(
                f"wrong image size ({got_w}x{got_h}, expected {expected_w}x{expected_h})"
            )

    fg_min = limits.get("fg_min")
    fg_max = limits.get("fg_max")
    fg_frac = float(measurement.get("fg_frac", 0.0))
    if fg_min is not None and fg_frac < float(fg_min):
        problems.append(
            f"part not visible / too little foreground "
            f"(fg_frac={fg_frac:.3f} < {float(fg_min):.3f})"
        )
    if fg_max is not None and fg_frac > float(fg_max):
        problems.append(
            f"frame too full / foreground out of range "
            f"(fg_frac={fg_frac:.3f} > {float(fg_max):.3f})"
        )

    return (not problems, "; ".join(problems))
