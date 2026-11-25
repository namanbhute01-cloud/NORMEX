#!/usr/bin/env python3
"""Camera Inspector CLI — capture / train / inspect / list / rollback / validate.

Every command works offline from a terminal.  The camera path uses OpenCV;
the file path uses the loader.  Both feed the same preprocessing and model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from module_0_common.config import config_hash, load_config, resolve_path
from module_1_acquisition.loader import list_images, load_frame
from module_2_preprocessing.pipeline import preprocess
from module_2_preprocessing.transform import map_to_source
from module_3_learning.backbone import Backbone
from module_3_learning.profile_store import ProfileError, ProfileStore
from module_3_learning.trainer import train
from module_4_inspection.anomaly_map import overlay
from module_4_inspection.inspector import Inspector
from module_5_decision.decide import decide


def _backbone(cfg: dict) -> Backbone:
    return Backbone(resolve_path(cfg["backbone"]["weights"]))


def _unique_heatmap_path(out_dir: Path, stem: str) -> Path:
    candidate = out_dir / f"{stem}_heatmap.png"
    index = 2
    while candidate.exists():
        candidate = out_dir / f"{stem}_{index}_heatmap.png"
        index += 1
    return candidate


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    backbone = _backbone(cfg)
    result = train(args.product, args.images, cfg, backbone, ProfileStore())
    print("\ntrain summary")
    print(f"  product       : {result.product_id}")
    print(f"  version       : {result.version}")
    print(f"  good images   : {result.n_good} (+{result.n_aug} augmented)")
    print(f"  t_lo / t_hi   : {result.calibration.t_lo:.3f} / {result.calibration.t_hi:.3f}")
    for stage, ms in result.stages_ms.items():
        print(f"  {stage:<14}: {ms / 1000.0:8.2f} s")
    print(f"  total         : {result.total_seconds:.2f} s")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    from module_1_acquisition.camera import (
        CameraError,
        capture_session,
        list_cameras,
        resolve_url,
    )

    if args.list:
        cameras = list_cameras(args.max_index)
        if not cameras:
            print("no usable local cameras found (use --ip/--url for a phone camera)", file=sys.stderr)
            return 1
        print("usable camera indexes:", ", ".join(str(index) for index in cameras))
        return 0

    if not args.out:
        print("capture: --out is required unless --list is used", file=sys.stderr)
        return 2

    url = args.url or ""
    snapshot_url = args.snapshot_url or ""
    if args.snapshot:
        if args.snapshot_url:
            print("capture: use --snapshot-url or --snapshot, not both", file=sys.stderr)
            return 2
        if args.ip:
            snapshot_url = resolve_url(args.ip, args.port, snapshot=True)
        elif args.url:
            snapshot_url, url = args.url, ""
        else:
            print("capture: --snapshot needs --ip or --url", file=sys.stderr)
            return 2
    elif args.ip:
        if args.url or args.snapshot_url:
            print("capture: use --ip or --url/--snapshot-url, not both", file=sys.stderr)
            return 2
        url = resolve_url(args.ip, args.port)
    if url and snapshot_url:
        print("capture: use --url or --snapshot-url, not both", file=sys.stderr)
        return 2

    try:
        paths = capture_session(
            args.out,
            index=args.camera,
            url=url,
            snapshot_url=snapshot_url,
            count=args.count,
            interval_s=args.interval,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
            preview=args.preview,
            countdown_s=args.countdown,
            prefix=args.prefix,
            timeout=args.timeout,
        )
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        return 2

    print(f"[capture] saved {len(paths)} frame(s) to {Path(args.out)}")
    print(f"[capture] next: python main.py train --product <id> --images {args.out}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    backbone = _backbone(cfg)
    store = ProfileStore()
    profile = store.load(
        args.product,
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )
    inspector = Inspector(backbone, profile, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    try:
        for path in list_images(args.image):
            started = time.perf_counter()
            frame = load_frame(path)
            prepared = preprocess(
                frame,
                cfg,
                quality_limits=profile.preprocess.get("quality_limits"),
                stored=profile.preprocess,
            )
            inspection = inspector.inspect(prepared) if prepared.quality_ok else None
            total_ms = (time.perf_counter() - started) * 1000.0
            verdict = decide(
                prepared,
                inspection,
                profile.calib,
                args.product,
                profile.version,
                total_ms,
            )
            verdict.preprocess_ms = prepared.latency_ms

            if inspection is not None and not args.no_heatmap:
                source_map = map_to_source(inspection.anomaly_map, prepared.geometry)
                roi_x, roi_y, roi_w, roi_h = prepared.geometry["roi"]
                mask = np.zeros(source_map.shape, dtype=np.uint8)
                mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = 255
                heatmap = overlay(
                    frame.image,
                    source_map,
                    vmin=0.0,
                    vmax=profile.calib.t_hi * 1.5,
                    mask=mask,
                )
                heatmap_path = _unique_heatmap_path(out_dir, path.stem)
                if not cv2.imwrite(str(heatmap_path), heatmap):
                    raise RuntimeError(f"could not write heatmap: {heatmap_path}")
                verdict.heatmap_path = str(heatmap_path)

            payload = json.dumps(asdict(verdict))
            print(payload)
            if jsonl_file is not None:
                jsonl_file.write(payload + "\n")
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = ProfileStore()
    products = store.list_products()
    if not products:
        print("no profiles found")
        return 0
    print(f"{'product':<32} {'current':<8} versions")
    for product in products:
        print(
            f"{product['product_id']:<32} {str(product['current']):<8} "
            f"{', '.join(product['versions'])}"
        )
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    store = ProfileStore()
    previous = store.rollback(args.product)
    print(f"rolled back {args.product} to {previous}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from scripts.run_validation import run

    if args.category == "all":
        from module_6_evaluation.mvtec import list_categories

        categories = list_categories(args.dataset)
    else:
        categories = [args.category]
    payload = run(args.dataset, categories, args.config, args.out, limit=args.limit)
    return 0 if payload["all_gates_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Camera Inspector (offline, CPU)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="learn a profile from good images only")
    p_train.add_argument("--product", required=True)
    p_train.add_argument("--images", required=True, help="folder of good images")
    p_train.add_argument("--config", default=None)
    p_train.set_defaults(func=cmd_train)

    p_capture = sub.add_parser("capture", help="capture from a laptop webcam or mobile IP camera")
    p_capture.add_argument("--out", default=None, help="folder for captured PNG frames")
    p_capture.add_argument("--camera", type=int, default=0, help="local camera index (default 0)")
    p_capture.add_argument("--url", default=None,
                           help="IP camera stream URL, e.g. http://192.168.1.5:8080/video")
    p_capture.add_argument("--snapshot-url", default=None,
                           help="JPEG snapshot URL, e.g. http://192.168.1.5:8080/shot.jpg")
    p_capture.add_argument("--ip", default=None,
                           help="phone LAN address; builds http://IP:PORT/video")
    p_capture.add_argument("--port", type=int, default=8080)
    p_capture.add_argument("--snapshot", action="store_true",
                           help="with --ip or --url, poll JPEG snapshots instead of a stream")
    p_capture.add_argument("--timeout", type=float, default=5.0,
                           help="snapshot HTTP timeout in seconds")
    p_capture.add_argument("--count", type=int, default=30, help="number of frames")
    p_capture.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    p_capture.add_argument("--width", type=int, default=1280)
    p_capture.add_argument("--height", type=int, default=720)
    p_capture.add_argument("--warmup", type=int, default=15, help="discard early frames")
    p_capture.add_argument("--countdown", type=float, default=3.0)
    p_capture.add_argument("--prefix", default="good")
    p_capture.add_argument("--preview", action="store_true", help="live window if OpenCV has GUI support")
    p_capture.add_argument("--list", action="store_true", help="list usable local camera indexes and exit")
    p_capture.add_argument("--max-index", type=int, default=5)
    p_capture.set_defaults(func=cmd_capture)

    p_inspect = sub.add_parser("inspect", help="inspect one image or a folder")
    p_inspect.add_argument("--product", required=True)
    p_inspect.add_argument("--image", required=True, help="image file or folder")
    p_inspect.add_argument("--out", default="results/inspect")
    p_inspect.add_argument("--jsonl", default=None, help="also write one JSON object per image")
    p_inspect.add_argument("--no-heatmap", action="store_true")
    p_inspect.add_argument("--config", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    p_list = sub.add_parser("list", help="list product profiles")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser("rollback", help="repoint a product to its previous version")
    p_rollback.add_argument("--product", required=True)
    p_rollback.set_defaults(func=cmd_rollback)

    p_validate = sub.add_parser("validate", help="run the MVTec AD benchmark")
    p_validate.add_argument("--dataset", default="datasets/mvtec_ad")
    p_validate.add_argument("--category", default="bottle", help="category name or 'all'")
    p_validate.add_argument("--config", default=None)
    p_validate.add_argument("--out", default="results")
    p_validate.add_argument("--limit", type=int, default=None)
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ProfileError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
