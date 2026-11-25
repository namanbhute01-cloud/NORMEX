"""M4: anomaly map construction, reduction and heatmap overlay."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def to_map(patch_scores: torch.Tensor | np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Upsample ``[h,w]`` patch scores to ``size``x``size`` and smooth.

    Smoothing suppresses single-cell noise; ``sigma`` is in output pixels.
    """

    if torch.is_tensor(patch_scores):
        array = patch_scores.detach().cpu().numpy()
    else:
        array = np.asarray(patch_scores)
    array = array.astype(np.float32, copy=False)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected [h,w] or [1,h,w] patch scores, got {array.shape}")

    resized = cv2.resize(array, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
    if sigma and sigma > 0:
        resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=float(sigma))
    return resized.astype(np.float32, copy=False)


def image_score(anomaly_map: np.ndarray, mode: str = "max", topk_frac: float = 0.01) -> float:
    """Reduce an anomaly map to one image-level score."""

    values = np.asarray(anomaly_map, dtype=np.float32).ravel()
    if values.size == 0:
        raise ValueError("empty anomaly map")
    if mode == "max":
        return float(values.max())
    if mode == "topk":
        k = max(1, int(round(values.size * float(topk_frac))))
        return float(np.partition(values, -k)[-k:].mean())
    raise ValueError(f"unknown image_score mode {mode!r} (expected 'max' or 'topk')")


def overlay(
    display_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    vmin: float = 0.0,
    vmax: float | None = None,
    alpha: float = 0.5,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """JET heatmap over the display image.

    ``vmax`` should come from the profile threshold (``t_hi * 1.5``) so the
    colours mean the same thing on every image; never normalize per image.
    ``mask`` (optional, same HxW) keeps pixels outside the ROI untouched.
    """

    if vmax is None:
        vmax = float(np.max(anomaly_map))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = np.clip((anomaly_map - vmin) / (vmax - vmin), 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(display_bgr, 1.0 - alpha, heat, alpha, 0.0)
    if mask is None:
        return blended

    weight = (np.asarray(mask) > 0).astype(np.float32)[..., None]
    return np.clip(
        display_bgr.astype(np.float32) * (1.0 - weight) + blended.astype(np.float32) * weight,
        0,
        255,
    ).astype(np.uint8)
