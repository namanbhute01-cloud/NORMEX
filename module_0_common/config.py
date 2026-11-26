"""YAML config loading, validation and hashing.

The config hash covers only the keys that change the meaning of extracted
features.  A profile trained under a different hash must refuse to load.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Every leaf key the application understands.  Unknown keys are an error:
# a typo must never silently fall back to a default.
_ALLOWED_LEAVES = {
    "seed",
    "backbone.weights",
    "input.size",
    "embedding.d",
    "embedding.pool",
    "algo.name",
    "algo.padim.eps",
    "train.augment_factor",
    "train.augment.shift",
    "train.augment.rot_deg",
    "train.augment.brightness",
    "train.augment.contrast",
    "train.augment.flip",
    "train.batch_size",
    "calibration.kfold",
    "calibration.sigma_k",
    "calibration.review_margin",
    "anomaly_map.sigma",
    "anomaly_map.image_score",
    "anomaly_map.topk_frac",
    "preprocess.keep_aspect",
    "preprocess.pad_value",
    "preprocess.roi.enabled",
    "preprocess.roi.x",
    "preprocess.roi.y",
    "preprocess.roi.width",
    "preprocess.roi.height",
    "preprocess.align.enabled",
    "preprocess.align.reference",
    "preprocess.align.max_shift_px",
    "preprocess.align.max_rotation_deg",
    "preprocess.denoise.enabled",
    "preprocess.denoise.method",
    "preprocess.denoise.ksize",
    "preprocess.denoise.d",
    "preprocess.denoise.sigma_color",
    "preprocess.denoise.sigma_space",
    "preprocess.illumination.enabled",
    "preprocess.illumination.method",
    "preprocess.illumination.sigma",
    "preprocess.illumination.gamma",
    "preprocess.illumination.clip_limit",
    "preprocess.illumination.tile_grid",
    "preprocess.illumination.target",
    "quality.enabled",
    "quality.check_foreground",
    "quality.require_same_size",
    "runtime.threads",
}

# Keys that determine feature compatibility between train and inspect.
_HASH_KEYS = (
    "input.size",
    "embedding.d",
    "embedding.pool",
    "backbone.weights",
    "algo.name",
    "preprocess.keep_aspect",
    "preprocess.pad_value",
    "preprocess.roi.enabled",
    "preprocess.roi.x",
    "preprocess.roi.y",
    "preprocess.roi.width",
    "preprocess.roi.height",
    "preprocess.align.enabled",
    "preprocess.align.reference",
    "preprocess.align.max_shift_px",
    "preprocess.align.max_rotation_deg",
    "preprocess.denoise.enabled",
    "preprocess.denoise.method",
    "preprocess.denoise.ksize",
    "preprocess.denoise.d",
    "preprocess.denoise.sigma_color",
    "preprocess.denoise.sigma_space",
    "preprocess.illumination.enabled",
    "preprocess.illumination.method",
    "preprocess.illumination.sigma",
    "preprocess.illumination.gamma",
    "preprocess.illumination.clip_limit",
    "preprocess.illumination.tile_grid",
    "preprocess.illumination.target",
)


def _flatten(node: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _require_int(flat: dict, key: str, lo: int, hi: int) -> None:
    value = flat[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config: {key} must be an integer, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise ValueError(f"config: {key} must be in [{lo}, {hi}], got {value}")


def _require_float(flat: dict, key: str, lo: float, hi: float) -> None:
    value = flat[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config: {key} must be a number, got {type(value).__name__}")
    if not lo <= float(value) <= hi:
        raise ValueError(f"config: {key} must be in [{lo}, {hi}], got {value}")


def validate_config(cfg: dict) -> None:
    """Raise ``ValueError`` on unknown keys, missing keys or bad ranges."""

    if not isinstance(cfg, dict):
        raise ValueError("config: top level must be a mapping")

    flat = _flatten(cfg)
    unknown = sorted(set(flat) - _ALLOWED_LEAVES)
    if unknown:
        raise ValueError(f"config: unknown key(s): {', '.join(unknown)}")
    missing = sorted(_ALLOWED_LEAVES - set(flat))
    if missing:
        raise ValueError(f"config: missing key(s): {', '.join(missing)}")

    _require_int(flat, "seed", 0, 2**31 - 1)
    _require_int(flat, "input.size", 64, 1024)
    if flat["input.size"] % 32 != 0:
        raise ValueError("config: input.size must be a multiple of 32 (ResNet strides)")
    _require_int(flat, "embedding.d", 1, 448)
    _require_int(flat, "embedding.pool", 1, 16)
    _require_float(flat, "algo.padim.eps", 1e-6, 10.0)
    _require_int(flat, "train.augment_factor", 0, 20)
    _require_float(flat, "train.augment.shift", 0.0, 0.2)
    _require_float(flat, "train.augment.rot_deg", 0.0, 15.0)
    _require_float(flat, "train.augment.brightness", 0.0, 0.5)
    _require_float(flat, "train.augment.contrast", 0.0, 0.5)
    _require_int(flat, "train.batch_size", 1, 256)
    _require_int(flat, "calibration.kfold", 2, 20)
    _require_float(flat, "calibration.sigma_k", 0.0, 10.0)
    _require_float(flat, "calibration.review_margin", 1.0, 3.0)
    _require_float(flat, "anomaly_map.sigma", 0.0, 50.0)
    _require_float(flat, "anomaly_map.topk_frac", 1e-6, 1.0)

    if not isinstance(flat["backbone.weights"], str) or not flat["backbone.weights"]:
        raise ValueError("config: backbone.weights must be a non-empty string")
    if flat["algo.name"] not in {"padim"}:
        raise ValueError(
            f"config: algo.name must be one of ['padim'] in this phase, got {flat['algo.name']!r}"
        )
    if flat["anomaly_map.image_score"] not in {"max", "topk"}:
        raise ValueError("config: anomaly_map.image_score must be 'max' or 'topk'")
    if not isinstance(flat["train.augment.flip"], bool):
        raise ValueError("config: train.augment.flip must be a boolean")
    if not isinstance(flat["quality.enabled"], bool):
        raise ValueError("config: quality.enabled must be a boolean")
    if not isinstance(flat["quality.check_foreground"], bool):
        raise ValueError("config: quality.check_foreground must be a boolean")
    if not isinstance(flat["quality.require_same_size"], bool):
        raise ValueError("config: quality.require_same_size must be a boolean")

    if not isinstance(flat["preprocess.keep_aspect"], bool):
        raise ValueError("config: preprocess.keep_aspect must be a boolean")
    _require_int(flat, "preprocess.pad_value", 0, 255)
    for key in (
        "preprocess.roi.enabled",
        "preprocess.align.enabled",
        "preprocess.denoise.enabled",
        "preprocess.illumination.enabled",
    ):
        if not isinstance(flat[key], bool):
            raise ValueError(f"config: {key} must be a boolean")
    _require_int(flat, "preprocess.roi.x", 0, 100_000)
    _require_int(flat, "preprocess.roi.y", 0, 100_000)
    _require_int(flat, "preprocess.roi.width", 0, 100_000)
    _require_int(flat, "preprocess.roi.height", 0, 100_000)
    if flat["preprocess.roi.enabled"] and (
        flat["preprocess.roi.width"] <= 0 or flat["preprocess.roi.height"] <= 0
    ):
        raise ValueError("config: enabled ROI needs positive width and height")
    if not isinstance(flat["preprocess.align.reference"], str):
        raise ValueError("config: preprocess.align.reference must be a string path")
    if flat["preprocess.align.enabled"] and not flat["preprocess.align.reference"]:
        raise ValueError("config: enabled alignment needs a reference good image path")
    _require_float(flat, "preprocess.align.max_shift_px", 0.0, 100.0)
    _require_float(flat, "preprocess.align.max_rotation_deg", 0.0, 10.0)
    if flat["preprocess.denoise.method"] not in {"median", "bilateral"}:
        raise ValueError("config: preprocess.denoise.method must be 'median' or 'bilateral'")
    _require_int(flat, "preprocess.denoise.ksize", 3, 15)
    if flat["preprocess.denoise.ksize"] % 2 == 0:
        raise ValueError("config: preprocess.denoise.ksize must be odd")
    _require_int(flat, "preprocess.denoise.d", 1, 15)
    _require_float(flat, "preprocess.denoise.sigma_color", 1.0, 100.0)
    _require_float(flat, "preprocess.denoise.sigma_space", 1.0, 100.0)
    if flat["preprocess.illumination.method"] not in {"background", "exposure", "gamma", "clahe"}:
        raise ValueError(
            "config: preprocess.illumination.method must be 'background', 'gamma' or 'clahe'"
        )
    _require_float(flat, "preprocess.illumination.sigma", 1.0, 200.0)
    _require_float(flat, "preprocess.illumination.gamma", 0.2, 5.0)
    _require_float(flat, "preprocess.illumination.clip_limit", 0.1, 10.0)
    _require_int(flat, "preprocess.illumination.tile_grid", 1, 32)
    threads = flat["runtime.threads"]
    if threads != "auto" and (not isinstance(threads, int) or threads < 1):
        raise ValueError("config: runtime.threads must be 'auto' or a positive integer")


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate a YAML config."""

    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Return a deep copy of ``cfg`` with dotted-path overrides applied."""

    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key not in _ALLOWED_LEAVES:
            raise ValueError(f"config override: unknown key {key!r}")
        _set(out, key, value)
    validate_config(out)
    return out


def config_hash(cfg: dict) -> str:
    """Short hash of the feature-defining config subset."""

    subset = {key: _get(cfg, key) for key in _HASH_KEYS}
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""

    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)
