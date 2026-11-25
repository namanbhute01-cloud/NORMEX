"""Typed contracts shared between modules.

The whole pipeline talks through these dataclasses only.  Swapping the
anomaly algorithm (e.g. PaDiM -> PatchCore) must not change them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Frame:
    """M1 -> M2.  One raw photograph."""

    image: np.ndarray  # HxWx3 uint8, BGR
    source: str


@dataclass
class Prepared:
    """M2 -> M3/M4.  A model-ready image plus its display copy."""

    tensor: torch.Tensor  # [1, 3, S, S] float32, ImageNet-normalized
    display: np.ndarray  # SxSx3 uint8, BGR (letterboxed model view)
    quality_ok: bool
    quality_msg: str = ""
    quality: Optional[dict] = None  # {"blur": float, "mean": float, "sat_frac": float}
    # Mapping from the SxS model view back to the original photograph:
    # source size, ROI box, letterbox scale/padding and optional alignment warp.
    geometry: Optional[dict] = None
    latency_ms: float = 0.0
    stages_ms: dict = field(default_factory=dict)


@dataclass
class Inspection:
    """M4 -> M5.  Raw anomaly measurement, before thresholding."""

    score: float  # image-level anomaly score
    anomaly_map: np.ndarray  # SxS float32, smoothed
    latency_ms: float  # backbone + score + map only
    stages_ms: dict = field(default_factory=dict)


@dataclass
class Calibration:
    """Per-product thresholds derived from held-out good images only."""

    t_lo: float
    t_hi: float
    good_scores: list[float]
    mu: float
    sd: float
    sigma_k: float
    review_margin: float
    kfold: int

    def to_dict(self) -> dict:
        return {
            "t_lo": float(self.t_lo),
            "t_hi": float(self.t_hi),
            "good_scores": [float(s) for s in self.good_scores],
            "mu": float(self.mu),
            "sd": float(self.sd),
            "sigma_k": float(self.sigma_k),
            "review_margin": float(self.review_margin),
            "kfold": int(self.kfold),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(
            t_lo=float(data["t_lo"]),
            t_hi=float(data["t_hi"]),
            good_scores=[float(s) for s in data["good_scores"]],
            mu=float(data["mu"]),
            sd=float(data["sd"]),
            sigma_k=float(data["sigma_k"]),
            review_margin=float(data["review_margin"]),
            kfold=int(data["kfold"]),
        )


@dataclass
class Verdict:
    """M5 -> caller / CLI."""

    product_id: str
    status: str  # GOOD | REVIEW | DEFECTIVE | BAD_IMAGE
    score: float
    score_norm: float  # score / t_lo (1.0 = edge of normal)
    t_lo: float
    t_hi: float
    latency_ms: float  # total, including preprocessing
    profile_version: str
    heatmap_path: Optional[str] = None
    message: str = ""
    preprocess_ms: float = 0.0  # M2 time, measured separately from M4
