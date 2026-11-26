#!/usr/bin/env python3
"""Measure end-to-end inspection latency (preprocess + model + map + decide).

Used to verify the < 1000 ms P95 gate on the actual target machine.

    python scripts/bench_latency.py --product screw --config configs/mvtec_fast.yaml
    python scripts/bench_latency.py --product screw --config configs/low_latency.yaml --threads 1
    python scripts/bench_latency.py --product demo --images datasets/demo/new_good --runs 50

Writes results/latency_<timestamp>.json so the dashboard can show it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_0_common.config import config_hash, load_config, resolve_path  # noqa: E402
from module_1_acquisition.loader import list_images, load_frame  # noqa: E402
from module_2_preprocessing.pipeline import preprocess  # noqa: E402
from module_3_learning.backbone import Backbone  # noqa: E402
from module_3_learning.profile_store import ProfileStore  # noqa: E402
from module_4_inspection.inspector import Inspector  # noqa: E402
from module_5_decision.decide import decide  # noqa: E402


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--product", required=True)
    parser.add_argument("--config", default="configs/mvtec_fast.yaml")
    parser.add_argument("--images", default=None, help="image folder/file (default: datasets/mvtec_ad/screw/test/good)")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=0, help="0 = config 'auto'")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    backbone = Backbone(resolve_path(cfg["backbone"]["weights"]))
    store = ProfileStore(ROOT / "profiles")
    profile = store.load(
        args.product,
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )
    inspector = Inspector(backbone, profile, cfg)

    if args.images:
        paths = list_images(args.images)
    else:
        paths = list_images(ROOT / "datasets" / "mvtec_ad" / "screw" / "test" / "good")
    if not paths:
        print("no images found", file=sys.stderr)
        return 2

    # warm-up
    for path in paths[:3]:
        prepared = preprocess(load_frame(path), cfg, stored=profile.preprocess, apply_quality=False)
        inspector.inspect(prepared)

    totals: list[float] = []
    prep_ms: list[float] = []
    model_ms: list[float] = []
    for i in range(args.runs):
        path = paths[i % len(paths)]
        t0 = time.perf_counter()
        frame = load_frame(path)
        prepared = preprocess(frame, cfg, stored=profile.preprocess, apply_quality=False)
        inspection = inspector.inspect(prepared)
        decide(prepared, inspection, profile.calib, args.product, profile.version,
               (time.perf_counter() - t0) * 1000.0)
        total = (time.perf_counter() - t0) * 1000.0
        totals.append(total)
        prep_ms.append(prepared.latency_ms)
        model_ms.append(inspection.latency_ms)

    record = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "product": args.product,
        "config": args.config,
        "input_size": int(cfg["input"]["size"]),
        "runs": args.runs,
        "threads": threads,
        "cpu_model": _cpu_model(),
        "latency_ms_mean": float(np.mean(totals)),
        "latency_ms_p50": float(np.percentile(totals, 50)),
        "latency_ms_p95": p95(totals),
        "preprocess_ms_mean": float(np.mean(prep_ms)),
        "model_ms_mean": float(np.mean(model_ms)),
        "gate_p95_under_1000ms": p95(totals) < 1000.0,
    }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"latency_{record['timestamp']}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps(record, indent=2))
    print(f"\nwrote {out_file}")
    return 0 if record["gate_p95_under_1000ms"] else 1


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
