#!/usr/bin/env python3
"""Train a profile for every good-only dataset under datasets/.

    python scripts/train_all.py                      # all missing profiles
    python scripts/train_all.py --force              # retrain everything
    python scripts/train_all.py --only pills,bottle  # subset

Dataset layout expected (produced by scripts/convert_datasets.py or the MVTec
archive)::

    datasets/<root>/<category>/train/good/*.png

Product naming matches the dashboard and the validation harness:

    datasets/mvtec_ad/screw  -> mvtec_screw
    datasets/extra/pills     -> pills
    datasets/demo/good       -> handled by launch.py train-demo (skipped here)

Existing profiles are skipped unless --force, so this is safe to re-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _python() -> str:
    for candidate in (
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def product_id(root: str, category: str) -> str:
    if root == "mvtec_ad":
        return f"mvtec_{category}"
    if root == "extra":
        return category
    return f"{root}_{category}"


def discover(datasets: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    if not datasets.is_dir():
        return rows
    for root_dir in sorted(p for p in datasets.iterdir() if p.is_dir()):
        for category_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
            good = category_dir / "train" / "good"
            if good.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in good.iterdir()):
                rows.append((root_dir.name, category_dir.name, good))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train every good-only dataset")
    parser.add_argument("--config", default="configs/mvtec_fast.yaml")
    parser.add_argument("--force", action="store_true", help="retrain existing profiles")
    parser.add_argument("--only", default="", help="comma-separated product ids")
    args = parser.parse_args(argv)

    selected = {s.strip() for s in args.only.split(",") if s.strip()}
    datasets = ROOT / "datasets"
    profiles = ROOT / "profiles"
    entries = discover(datasets)
    if not entries:
        print("[train-all] no dataset with train/good found under datasets/", file=sys.stderr)
        return 2

    planned = []
    for root, category, good in entries:
        product = product_id(root, category)
        if selected and product not in selected:
            continue
        if (profiles / product / "current").is_file() and not args.force:
            print(f"[train-all] skip {product}: profile already exists (use --force to retrain)")
            continue
        planned.append((product, good))

    if not planned:
        print("[train-all] nothing to train")
        return 0

    python = _python()
    failures = 0
    try:
        for product, good in planned:
            count = sum(1 for p in good.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            print(f"\n[train-all] {product}: {count} good images from {good.relative_to(ROOT)}")
            command = [
                python, str(ROOT / "main.py"), "train",
                "--product", product,
                "--images", str(good),
                "--config", args.config,
            ]
            code = subprocess.call(command, cwd=str(ROOT))
            if code != 0:
                failures += 1
                print(f"[train-all] FAILED {product} (exit {code})", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[train-all] interrupted")
        return 130

    print(f"\n[train-all] done: {len(planned) - failures} trained, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
