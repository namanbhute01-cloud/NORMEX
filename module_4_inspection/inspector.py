"""M4: turn a prepared photograph into an anomaly score + heatmap."""

from __future__ import annotations

import time

import torch

from module_0_common.types import Inspection, Prepared
from module_3_learning.embedding import embed
from module_4_inspection.anomaly_map import image_score, to_map


class Inspector:
    """Backbone + profile loaded once, reused for every photograph."""

    def __init__(self, backbone: torch.nn.Module, profile, cfg: dict) -> None:
        self.backbone = backbone
        self.profile = profile
        self.size = profile.input_size
        self.channel_idx = profile.channel_idx
        self.pool = profile.pool
        self.sigma = float(cfg["anomaly_map"]["sigma"])
        self.mode = str(cfg["anomaly_map"]["image_score"])
        self.topk_frac = float(cfg["anomaly_map"]["topk_frac"])

    @torch.inference_mode()
    def inspect(self, prepared: Prepared) -> Inspection:
        """Run the full M4 pipeline on one prepared image."""

        if not prepared.quality_ok:
            raise ValueError(
                "Inspector received an image that failed the quality gate; "
                "check Prepared.quality_ok before calling inspect()"
            )
        if prepared.tensor.shape[-2:] != (self.size, self.size):
            raise ValueError(
                f"prepared tensor is {tuple(prepared.tensor.shape[-2:])}, "
                f"profile expects {(self.size, self.size)}"
            )

        started = time.perf_counter()
        stages: dict[str, float] = {}

        t0 = time.perf_counter()
        features = self.backbone(prepared.tensor)
        embedding = embed(features, self.channel_idx, self.pool)
        stages["backbone"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        patch_scores = self.profile.algo.score(embedding)
        stages["score"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        anomaly_map = to_map(patch_scores, size=self.size, sigma=self.sigma)
        score = image_score(anomaly_map, mode=self.mode, topk_frac=self.topk_frac)
        stages["map"] = (time.perf_counter() - t0) * 1000.0

        return Inspection(
            score=float(score),
            anomaly_map=anomaly_map,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            stages_ms=stages,
        )
