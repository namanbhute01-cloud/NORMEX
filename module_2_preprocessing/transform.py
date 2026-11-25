"""M2: geometry-preserving resize, tensor conversion and heatmap mapping.

The model input is square, but photographs rarely are.  Stretching a 16:9
frame to 256x256 changes the shape of every feature; this module letterboxes
instead and records the exact inverse mapping so the anomaly heatmap can be
drawn back on the original photograph.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from module_0_common.types import Frame, Prepared

# ImageNet statistics, RGB order (torchvision convention).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(
    img_bgr: np.ndarray,
    size: int,
    pad_value: int = 0,
    keep_aspect: bool = True,
) -> tuple[np.ndarray, dict]:
    """Resize to ``size``x``size``, preserving aspect ratio when requested.

    Returns the image and the geometry needed to map a model-space anomaly
    map back to this image.
    """

    height, width = img_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image shape {img_bgr.shape}")
    if size <= 0:
        raise ValueError(f"invalid input size {size}")

    if keep_aspect:
        scale = min(float(size) / width, float(size) / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        out = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
        out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    else:
        new_w = new_h = int(size)
        pad_x = pad_y = 0
        out = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

    geometry = {
        "src_w": int(width),
        "src_h": int(height),
        "size": int(size),
        "keep_aspect": bool(keep_aspect),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
    }
    return out, geometry


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> ``[1,3,H,W]`` float32 ImageNet-normalized tensor."""

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return (
        torch.from_numpy(normalized.transpose(2, 0, 1))
        .unsqueeze(0)
        .contiguous()
        .float()
    )


def map_to_source(anomaly_map: np.ndarray, geometry: dict | None) -> np.ndarray:
    """Map a model-space ``SxS`` anomaly map back to the original photograph.

    The result is ``source_h x source_w``; pixels outside the ROI are zero.
    Alignment, when used, is inverted here so the heatmap lands on the
    photographed part, not on the aligned reference frame.
    """

    array = np.asarray(anomaly_map, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D anomaly map, got {array.shape}")
    if not geometry:
        return array

    size = int(geometry["size"])
    if array.shape != (size, size):
        array = cv2.resize(array, (size, size), interpolation=cv2.INTER_LINEAR)

    x, y, roi_w, roi_h = (int(v) for v in geometry["roi"])
    pad_x = int(geometry.get("pad_x", 0))
    pad_y = int(geometry.get("pad_y", 0))
    new_w = int(geometry.get("new_w", size))
    new_h = int(geometry.get("new_h", size))
    content = array[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    roi_map = cv2.resize(content, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    align_info = geometry.get("align") or {}
    warp = align_info.get("warp")
    if align_info.get("applied") and warp is not None:
        # ECC returns a warp from reference coordinates to input coordinates
        # (it is used with WARP_INVERSE_MAP to align the input).  Applying it
        # directly here maps an aligned-space heatmap back onto the input.
        roi_map = cv2.warpAffine(
            roi_map,
            np.asarray(warp, dtype=np.float32),
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    source = np.zeros((int(geometry["src_h"]), int(geometry["src_w"])), dtype=np.float32)
    source[y : y + roi_h, x : x + roi_w] = roi_map
    return source


def prepare(
    frame: Frame,
    size: int,
    quality_cfg: dict | None = None,
    quality_limits: dict | None = None,
    preprocess_cfg: dict | None = None,
) -> Prepared:
    """Backward-compatible wrapper around :func:`pipeline.preprocess`.

    New code should call ``module_2_preprocessing.pipeline.preprocess`` with
    the full application config so ROI/alignment/photometric settings are
    applied consistently at train and inspect time.
    """

    from module_2_preprocessing.pipeline import default_preprocess_cfg, preprocess

    cfg = {
        "input": {"size": int(size)},
        "quality": quality_cfg or {"enabled": False},
        "preprocess": preprocess_cfg or default_preprocess_cfg(),
    }
    return preprocess(
        frame,
        cfg,
        quality_limits=quality_limits,
        apply_quality=quality_limits is not None,
    )
