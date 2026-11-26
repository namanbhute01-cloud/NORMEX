#!/usr/bin/env python3
"""MVTec AD validation harness.

Trains on ``train/good`` only, inspects every test image, and writes
``results/mvtec_<timestamp>.json`` + ``.csv`` plus a pass/fail gate summary.

The dataset is never downloaded here; place it under ``datasets/mvtec_ad``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module_0_common.config import (  # noqa: E402
    PROJECT_ROOT,
    apply_overrides,
    config_hash,
    load_config,
    resolve_path,
)
from module_0_common.types import Verdict  # noqa: E402
from module_6_evaluation import metrics, mvtec  # noqa: E402
from module_1_acquisition.loader import load_frame  # noqa: E402
from module_2_preprocessing.pipeline import preprocess  # noqa: E402
from module_2_preprocessing.transform import map_to_source  # noqa: E402
from module_3_learning.backbone import Backbone  # noqa: E402
from module_3_learning.profile_store import ProfileStore  # noqa: E402
from module_3_learning.trainer import train  # noqa: E402
from module_4_inspection.inspector import Inspector  # noqa: E402
from module_5_decision.decide import decide  # noqa: E402

MAX_TRAIN_SECONDS = 600.0
MAX_P95_LATENCY_MS = 1000.0
MIN_IMAGE_AUROC = 0.95
MAX_MISSED_DEFECT_RATE = 0.10   # FAR here = defective image marked GOOD
MAX_FALSE_REJECT_RATE = 0.05    # FRR = good image marked DEFECTIVE
BOTTLE_AUROC_GATE = 0.97


def _profile_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024.0 * 1024.0)


def evaluate_category(
    category: str,
    dataset_root: Path,
    base_cfg: dict,
    backbone: Backbone,
    store: ProfileStore,
    out_dir: Path,
    limit: int | None = None,
) -> dict:
    """Train + inspect one MVTec category.  Returns a metrics record."""

    cfg = apply_overrides(deepcopy(base_cfg), {"quality.enabled": False})
    train_dir = mvtec.train_good_dir(dataset_root, category)
    test_items = mvtec.list_test(dataset_root, category)
    if limit:
        test_items = test_items[:limit]

    product_id = f"mvtec_{category}"
    print(f"\n=== {category}: training on {train_dir} ===")
    result = train(product_id, train_dir, cfg, backbone, store)
    profile = store.load(
        product_id,
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )
    inspector = Inspector(backbone, profile, cfg)

    # Warm up 3 inferences (excluded from timing).
    warm = preprocess(
        load_frame(test_items[0].path),
        cfg,
        stored=profile.preprocess,
        apply_quality=False,
    )
    for _ in range(3):
        inspector.inspect(warm)

    scores: list[float] = []
    statuses: list[str] = []
    labels: list[int] = []
    latencies: list[float] = []
    preprocess_ms: list[float] = []
    read_ms: list[float] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    print(f"[{category}] inspecting {len(test_items)} test images")
    for item in test_items:
        t0 = time.perf_counter()
        frame = load_frame(item.path)
        read_elapsed = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        prepared = preprocess(
            frame,
            cfg,
            stored=profile.preprocess,
            apply_quality=False,  # MVTec harness measures the model, not the gate
        )
        inspection = inspector.inspect(prepared)
        verdict: Verdict = decide(
            prepared,
            inspection,
            profile.calib,
            product_id,
            profile.version,
            (time.perf_counter() - t1) * 1000.0,
        )

        scores.append(verdict.score)
        statuses.append(verdict.status)
        labels.append(item.label)
        latencies.append(verdict.latency_ms)
        preprocess_ms.append(prepared.latency_ms)
        read_ms.append(read_elapsed)
        if item.mask_path is not None:
            maps.append(map_to_source(inspection.anomaly_map, prepared.geometry))
            masks.append(mvtec.load_mask(item.mask_path, None))

    rates = metrics.far_frr(statuses, labels)
    record = {
        "category": category,
        "train_seconds": float(result.total_seconds),
        "train_stages_ms": result.stages_ms,
        "n_train": int(result.n_good),
        "n_aug": int(result.n_aug),
        "n_test_good": int(sum(1 for label in labels if label == 0)),
        "n_test_defect": int(sum(1 for label in labels if label == 1)),
        "image_auroc": metrics.image_auroc(labels, scores),
        "pixel_auroc": metrics.pixel_auroc(maps, masks) if maps else float("nan"),
        "far": rates["far"],
        "frr": rates["frr"],
        "review_rate_good": rates["review_rate_good"],
        "review_rate_defect": rates["review_rate_defect"],
        "latency_ms_mean": float(np.mean(latencies)) if latencies else float("nan"),
        "latency_ms_p95": metrics.p95(latencies),
        "preprocess_ms_mean": float(np.mean(preprocess_ms)) if preprocess_ms else float("nan"),
        "preprocess_ms_p95": metrics.p95(preprocess_ms),
        "read_ms_mean": float(np.mean(read_ms)) if read_ms else float("nan"),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "profile_size_mb": _profile_size_mb(profile.path),
        "cpu_model": result.report.get("cpu_model", ""),
        "n_threads": int(result.report.get("n_threads", 0)),
        "torch_version": str(result.report.get("torch_version", "")),
        "version": profile.version,
    }

    gates = {
        "train_seconds<=600": record["train_seconds"] <= MAX_TRAIN_SECONDS,
        "latency_p95<1000ms": record["latency_ms_p95"] < MAX_P95_LATENCY_MS,
        "image_auroc>=0.95": record["image_auroc"] >= MIN_IMAGE_AUROC,
    }
    if not math.isnan(record["far"]):
        gates["missed_defects<=0.10"] = record["far"] <= MAX_MISSED_DEFECT_RATE
    if not math.isnan(record["frr"]):
        gates["false_rejects<=0.05"] = record["frr"] <= MAX_FALSE_REJECT_RATE
    if category == "bottle":
        gates["bottle_auroc>=0.97"] = record["image_auroc"] >= BOTTLE_AUROC_GATE
    record["gates"] = gates
    record["gates_pass"] = all(gates.values())

    print(
        f"[{category}] AUROC={record['image_auroc']:.4f} "
        f"pixelAUROC={record['pixel_auroc']:.4f} FAR={record['far']:.3f} "
        f"FRR={record['frr']:.3f} train={record['train_seconds']:.1f}s "
        f"p95={record['latency_ms_p95']:.1f}ms"
    )
    return record


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def run(
    dataset: str | Path,
    categories: list[str],
    config_path: str | Path | None,
    out_dir: str | Path,
    limit: int | None = None,
) -> dict:
    dataset_root = Path(dataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(config_path)
    backbone = Backbone(resolve_path(base_cfg["backbone"]["weights"]))
    store = ProfileStore(PROJECT_ROOT / "profiles")

    records = []
    for category in categories:
        records.append(
            evaluate_category(
                category, dataset_root, base_cfg, backbone, store, out_dir, limit=limit
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "dataset": str(dataset_root),
        "environment": {
            "cpu_model": records[0].get("cpu_model", "") if records else "",
            "n_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "categories": records,
        "all_gates_pass": all(r["gates_pass"] for r in records),
    }
    json_path = out_dir / f"mvtec_{timestamp}.json"
    csv_path = out_dir / f"mvtec_{timestamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    columns = [
        "category",
        "n_train",
        "n_test_good",
        "n_test_defect",
        "train_seconds",
        "image_auroc",
        "pixel_auroc",
        "far",
        "frr",
        "review_rate_good",
        "review_rate_defect",
        "latency_ms_mean",
        "latency_ms_p95",
        "preprocess_ms_mean",
        "preprocess_ms_p95",
        "peak_rss_mb",
        "profile_size_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print("\ncategory        AUROC    pixelAUROC  FAR     FRR     train_s  p95_ms")
    for record in records:
        print(
            f"{record['category']:<15} {_fmt(record['image_auroc']):>7} "
            f"{_fmt(record['pixel_auroc']):>11} {_fmt(record['far']):>7} "
            f"{_fmt(record['frr']):>7} {record['train_seconds']:>8.1f} "
            f"{record['latency_ms_p95']:>7.1f}"
        )
    print(f"\nwrote {json_path}\nwrote {csv_path}")
    print(f"all gates pass: {payload['all_gates_pass']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MVTec AD validation harness")
    parser.add_argument("--dataset", default="datasets/mvtec_ad")
    parser.add_argument("--category", default="bottle", help="category name or 'all'")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, default=None, help="max test images per category")
    args = parser.parse_args(argv)

    if args.category == "all":
        categories = mvtec.list_categories(args.dataset)
    else:
        categories = [args.category]

    payload = run(args.dataset, categories, args.config, args.out, limit=args.limit)
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
