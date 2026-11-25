"""M3: orchestrate good-only learning into a saved, calibrated profile.

Pipeline (each stage is timed and written to ``train_report.json``):

1. list + load good images (defect/test paths are structurally refused)
2. measure quality, learn the profile's quality limits
3. embed originals with the frozen backbone
4. embed defect-safe augmented copies
5. k-fold calibration on held-out good images
6. final fit on all originals + augmented copies
7. save a new profile version
"""

from __future__ import annotations

import os
import platform
import resource
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch

from module_0_common.config import config_hash
from module_0_common.seed import seed_everything
from module_0_common.timing import Stages
from module_0_common.types import Calibration, Frame
from module_1_acquisition.loader import list_images, load_frame
from module_2_preprocessing import augment, quality
from module_2_preprocessing import pipeline
from module_2_preprocessing.transform import IMAGENET_MEAN, IMAGENET_STD
from module_3_learning.backbone import Backbone
from module_3_learning.embedding import embed, make_channel_idx
from module_3_learning.profile_store import ProfileData, ProfileStore
from module_3_learning.registry import get_algo
from module_5_decision.calibrate import kfold_good_scores, make_calibration

MIN_GOOD_IMAGES = 10


@dataclass
class TrainResult:
    product_id: str
    version: str
    n_good: int
    n_aug: int
    calibration: Calibration
    stages_ms: dict
    total_seconds: float
    report: dict = field(default_factory=dict)


def _guard_training_path(path: Path) -> None:
    """Refuse anything that looks like a test / ground-truth folder."""

    for part in path.resolve().parts:
        lowered = part.lower()
        if lowered == "test" or "ground_truth" in lowered:
            raise ValueError(
                f"training path looks like a test/defect folder ({path}); "
                "training must use good images only"
            )


def _configure_threads(cfg: dict) -> int:
    configured = cfg["runtime"]["threads"]
    if configured == "auto":
        threads = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    else:
        threads = int(configured)
    torch.set_num_threads(max(1, threads))
    return max(1, threads)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _embed_frames(
    frames: list[Frame],
    backbone: Backbone,
    cfg: dict,
    channel_idx: torch.Tensor,
    pool: int,
    batch_size: int,
) -> torch.Tensor:
    """Prepare + embed frames in batches.  Returns [N,d,H,W] on CPU."""

    outputs: list[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        tensors = [
            pipeline.preprocess(frame, cfg, apply_quality=False).tensor
            for frame in batch_frames
        ]
        batch = torch.cat(tensors, dim=0)
        with torch.inference_mode():
            features = backbone(batch)
            embeddings = embed(features, channel_idx, pool)
        outputs.append(embeddings)
    return torch.cat(outputs, dim=0)


def train(
    product_id: str,
    image_dir: str | Path,
    cfg: dict,
    backbone: Backbone,
    store: Optional[ProfileStore] = None,
) -> TrainResult:
    """Learn a product profile from a directory of good images only."""

    started = time.perf_counter()
    store = store or ProfileStore()
    image_dir = Path(image_dir)
    _guard_training_path(image_dir)

    seed_everything(int(cfg["seed"]))
    threads = _configure_threads(cfg)

    stages = Stages()
    with stages.stage("load"):
        paths = list_images(image_dir)
        if len(paths) < MIN_GOOD_IMAGES:
            raise ValueError(
                f"need at least {MIN_GOOD_IMAGES} good images, found {len(paths)} in {image_dir}"
            )
        frames = [load_frame(path) for path in paths]
    print(f"[train] loaded {len(frames)} good images from {image_dir}")

    size = int(cfg["input"]["size"])
    d = int(cfg["embedding"]["d"])
    pool = int(cfg["embedding"]["pool"])
    channel_idx = make_channel_idx(total=sum(Backbone.out_channels), d=d, seed=int(cfg["seed"]))

    with stages.stage("quality"):
        measures = [pipeline.measure_quality(frame, cfg) for frame in frames]
        quality_limits = (
            quality.learn_limits(
                measures,
                check_foreground=bool(cfg["quality"]["check_foreground"]),
                require_same_size=bool(cfg["quality"]["require_same_size"]),
            )
            if cfg["quality"]["enabled"]
            else None
        )
    print(f"[train] quality limits: {quality_limits}")

    batch_size = int(cfg["train"]["batch_size"])

    with stages.stage("embed_orig"):
        E_orig = _embed_frames(frames, backbone, cfg, channel_idx, pool, batch_size)
    print(f"[train] original embeddings: {tuple(E_orig.shape)}")

    augment_factor = int(cfg["train"]["augment_factor"])
    E_aug: Optional[torch.Tensor] = None
    aug_index: list[list[int]] = [[] for _ in frames]
    if augment_factor > 0:
        with stages.stage("augment"):
            augmented_frames: list[Frame] = []
            rng = np.random.default_rng(int(cfg["seed"]) + 1)
            for image_index, frame in enumerate(frames):
                for copy_index in range(augment_factor):
                    augmented = augment.augment(frame.image, rng, cfg["train"]["augment"])
                    aug_index[image_index].append(len(augmented_frames))
                    augmented_frames.append(
                        Frame(image=augmented, source=f"{frame.source}#aug{copy_index}")
                    )
        with stages.stage("embed_aug"):
            E_aug = _embed_frames(augmented_frames, backbone, cfg, channel_idx, pool, batch_size)
        print(f"[train] augmented embeddings: {tuple(E_aug.shape)}")
    else:
        print("[train] augmentation disabled")

    kfold = min(int(cfg["calibration"]["kfold"]), len(frames))
    with stages.stage("calibrate"):
        good_scores = kfold_good_scores(
            E_orig,
            aug_index,
            E_aug,
            algo_factory=lambda: get_algo(cfg["algo"]["name"], cfg["algo"]),
            k=kfold,
            seed=int(cfg["seed"]),
            size=size,
            sigma=float(cfg["anomaly_map"]["sigma"]),
            mode=str(cfg["anomaly_map"]["image_score"]),
            topk_frac=float(cfg["anomaly_map"]["topk_frac"]),
        )
        calibration = make_calibration(
            good_scores,
            sigma_k=float(cfg["calibration"]["sigma_k"]),
            review_margin=float(cfg["calibration"]["review_margin"]),
            k=kfold,
        )
    print(
        f"[train] calibration: t_lo={calibration.t_lo:.3f} "
        f"t_hi={calibration.t_hi:.3f} (k={kfold})"
    )

    with stages.stage("fit"):
        algo = get_algo(cfg["algo"]["name"], cfg["algo"])
        fit_embeddings = E_orig if E_aug is None else torch.cat([E_orig, E_aug], dim=0)
        algo.fit(fit_embeddings)

    meta = {
        "algo": algo.name,
        "seed": int(cfg["seed"]),
        "input_size": size,
        "d": d,
        "pool": pool,
        "hw": list(algo.hw),
        "eps": float(cfg["algo"]["padim"]["eps"]),
        "backbone_sha256": backbone.sha256,
        "config_hash": config_hash(cfg),
        "n_good": len(frames),
        "n_aug": 0 if E_aug is None else int(E_aug.shape[0]),
        "kfold": kfold,
    }
    preprocess = {
        "size": size,
        "quality_enabled": bool(cfg["quality"]["enabled"]),
        "quality_limits": quality_limits,
        "normalize": {
            "mean": [float(v) for v in IMAGENET_MEAN],
            "std": [float(v) for v in IMAGENET_STD],
        },
    }
    preprocess.update(deepcopy(cfg["preprocess"]))

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report = {
        "product_id": product_id,
        "stages_ms": dict(stages.ms),
        "n_good": len(frames),
        "n_aug": 0 if E_aug is None else int(E_aug.shape[0]),
        "kfold": kfold,
        "cpu_model": _cpu_model(),
        "n_threads": threads,
        "torch_version": torch.__version__,
        "peak_rss_mb": float(peak_rss_mb),
        "profile_bytes": int(algo.nbytes + channel_idx.numel() * 8),
        "seed": int(cfg["seed"]),
    }

    with stages.stage("save"):
        version = store.save(
            product_id,
            ProfileData(
                algo=algo,
                channel_idx=channel_idx,
                calib=calibration,
                preprocess=preprocess,
                meta=meta,
                train_report=report,
            ),
        )
    report["stages_ms"] = dict(stages.ms)
    report["total_seconds"] = time.perf_counter() - started
    version_dir = store.root / product_id / version
    report["profile_bytes"] = int(
        sum(p.stat().st_size for p in version_dir.rglob("*") if p.is_file())
    )
    store.update_train_report(product_id, version, report)

    print(
        f"[train] saved {product_id}/{version} in "
        f"{report['total_seconds']:.1f}s (peak RSS {peak_rss_mb:.0f} MB)"
    )

    return TrainResult(
        product_id=product_id,
        version=version,
        n_good=len(frames),
        n_aug=0 if E_aug is None else int(E_aug.shape[0]),
        calibration=calibration,
        stages_ms=dict(stages.ms),
        total_seconds=report["total_seconds"],
        report=report,
    )
