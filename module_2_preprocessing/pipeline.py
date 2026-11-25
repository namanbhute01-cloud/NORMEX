"""M2: the complete preprocessing pipeline.

Order (matching the blueprint):

    validate -> ROI -> optional alignment -> optional denoise
    -> optional illumination correction -> letterbox/resize
    -> BGR->RGB -> ImageNet normalization -> tensor

The same function is used for training and inspection.  Geometry is returned
with the tensor so M4's heatmap can be mapped back onto the original photo.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from module_0_common.config import resolve_path
from module_0_common.types import Frame, Prepared
from module_2_preprocessing import align, denoise, illumination, quality, roi
from module_2_preprocessing.transform import letterbox, to_tensor


def default_preprocess_cfg() -> dict:
    """Defaults used when a caller has no full application config."""

    return {
        "keep_aspect": True,
        "pad_value": 0,
        "roi": {
            "enabled": False,
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
        },
        "align": {
            "enabled": False,
            "reference": "",
            "max_shift_px": 8.0,
            "max_rotation_deg": 2.0,
        },
        "denoise": {
            "enabled": False,
            "method": "median",
            "ksize": 3,
            "d": 5,
            "sigma_color": 25.0,
            "sigma_space": 25.0,
        },
        "illumination": {
            "enabled": False,
            "method": "background",
            "sigma": 40.0,
            "gamma": 1.0,
            "clip_limit": 2.0,
            "tile_grid": 8,
        },
    }


def _merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _preprocess_cfg(cfg: dict, stored: dict | None) -> dict:
    pp = _merge(default_preprocess_cfg(), cfg.get("preprocess"))
    if stored:
        # A loaded profile is authoritative: inspect must reproduce training.
        pp = _merge(pp, {k: v for k, v in stored.items() if k in pp})
    return pp


def measure_quality(frame: Frame, cfg: dict, stored: dict | None = None) -> dict:
    """Measure quality on the ROI, using the same ROI as training/inspection."""

    pp = _preprocess_cfg(cfg, stored)
    height, width = frame.image.shape[:2]
    box = roi.resolve_box(pp.get("roi"), width, height)
    cropped = roi.crop(frame.image, box)
    return quality.measure(cropped)


def preprocess(
    frame: Frame,
    cfg: dict,
    *,
    quality_limits: dict | None = None,
    stored: dict | None = None,
    apply_quality: bool = True,
    reference: np.ndarray | None = None,
) -> Prepared:
    """Run the full M2 pipeline on one frame.

    ``stored`` is the ``preprocess.json`` of a loaded profile.  When present,
    its ROI/size/alignment/photometric settings win, so a profile can never
    be inspected with a different geometry than it was trained with.
    """

    started = time.perf_counter()
    stages: dict[str, float] = {}
    image = frame.image

    problems = quality.validate(image)
    if problems:
        raise ValueError(f"invalid frame from {frame.source}: {'; '.join(problems)}")

    pp = _preprocess_cfg(cfg, stored)
    size = int(stored["size"]) if stored and "size" in stored else int(cfg["input"]["size"])
    qcfg = cfg.get("quality") or {"enabled": False}
    if stored and "quality_enabled" in stored:
        quality_enabled = bool(stored["quality_enabled"])
    else:
        quality_enabled = bool(qcfg.get("enabled", True))
    if quality_limits is None and stored:
        quality_limits = stored.get("quality_limits")

    source_h, source_w = image.shape[:2]

    t0 = time.perf_counter()
    box = roi.resolve_box(pp.get("roi"), source_w, source_h)
    image = roi.crop(image, box)
    stages["roi"] = (time.perf_counter() - t0) * 1000.0

    measurement = None
    quality_ok = True
    quality_msg = ""
    if quality_enabled or (apply_quality and quality_limits is not None):
        t0 = time.perf_counter()
        measurement = quality.measure(image)
        stages["quality"] = (time.perf_counter() - t0) * 1000.0
        if apply_quality and quality_enabled and quality_limits is not None:
            quality_ok, quality_msg = quality.check(measurement, quality_limits)

    align_info: dict = {
        "ok": True,
        "applied": False,
        "cc": 1.0,
        "shift_px": 0.0,
        "rotation_deg": 0.0,
        "warp": None,
        "message": "",
    }
    if bool(pp["align"].get("enabled", False)):
        t0 = time.perf_counter()
        reference_image = reference
        if reference_image is None:
            reference_path = str(pp["align"].get("reference", "")).strip()
            if not reference_path:
                raise ValueError("alignment is enabled but no reference image is configured")
            reference_image = align.load_reference(resolve_path(reference_path))
        reference_image = roi.crop(reference_image, box)
        image, align_info = align.align(image, reference_image, pp["align"])
        stages["align"] = (time.perf_counter() - t0) * 1000.0
        if not align_info["ok"]:
            quality_ok = False
            quality_msg = align_info["message"]

    if bool(pp["denoise"].get("enabled", False)):
        t0 = time.perf_counter()
        image = denoise.apply(image, pp["denoise"])
        stages["denoise"] = (time.perf_counter() - t0) * 1000.0

    if bool(pp["illumination"].get("enabled", False)):
        t0 = time.perf_counter()
        image = illumination.apply(image, pp["illumination"])
        stages["illumination"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    model_view, geometry = letterbox(
        image,
        size,
        pad_value=int(pp.get("pad_value", 0)),
        keep_aspect=bool(pp.get("keep_aspect", True)),
    )
    geometry["roi"] = box
    geometry["align"] = align_info
    tensor = to_tensor(model_view)
    stages["tensor"] = (time.perf_counter() - t0) * 1000.0

    return Prepared(
        tensor=tensor,
        display=model_view,
        quality_ok=quality_ok,
        quality_msg=quality_msg,
        quality=measurement,
        geometry=geometry,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        stages_ms=stages,
    )
