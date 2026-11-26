#!/usr/bin/env python3
"""Convert the extra Kaggle datasets into NORMEX's good-only layout.

NORMEX trains only on good images, so any classification/detection dataset can
be used as a source of *good* examples.  The converter produces::

    datasets/extra/<name>/
    ├── train/good/     training images (good only)
    └── test/good/      held-out good images

Supported inputs:

* YOLO detection sets  (train/images + train/labels, valid/images or test/images)
  All images are treated as good.  Images whose label file is non-empty are
  additionally copied to ``test/defect/``; they are never used for training.
  (A detection box means "part present", not "part defective", so this is only
  a convenience — real defect evaluation needs real defect labels.)
* Plain image folders   (every image is good; a deterministic split is made).

Usage:
    python scripts/convert_datasets.py --pills ~/datasets/mvtec_ad/pills
    python scripts/convert_datasets.py --folder nut=/path/to/nut_images
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "datasets" / "extra"


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _copy(src: Path, dst_dir: Path, prefix: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}{src.name}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def convert_yolo(name: str, src: Path, max_train: int, max_test: int, seed: int = 0) -> dict:
    """Convert a YOLO detection set (labels ignored for training)."""

    train_dir = src / "train" / "images"
    test_dir = src / "valid" / "images"
    if not test_dir.is_dir():
        test_dir = src / "test" / "images"
    label_dir = src / "train" / "labels"

    train_imgs = _images(train_dir)[:max_train]
    test_imgs = _images(test_dir)[:max_test]
    if not train_imgs and not test_imgs:
        raise SystemExit(f"[convert] {name}: no images found under {train_dir} / {test_dir}")

    out = OUT_ROOT / name
    n_train = n_test = n_defect = 0
    for img in train_imgs:
        _copy(img, out / "train" / "good", "good_")
        n_train += 1
    for img in test_imgs:
        _copy(img, out / "test" / "good", "good_")
        n_test += 1
        label = label_dir / f"{img.stem}.txt"
        if label.is_file() and label.stat().st_size > 0:
            _copy(img, out / "test" / "defect", "boxed_")
            n_defect += 1

    return {"name": name, "train_good": n_train, "test_good": n_test, "test_defect": n_defect,
            "note": "YOLO detection set; boxes are not defects"}


def convert_folder(name: str, src: Path, max_train: int, max_test: int, seed: int = 0) -> dict:
    """Convert a plain image folder: deterministic split into train/test good."""

    images = _images(src)
    if not images:
        raise SystemExit(f"[convert] {name}: no images found in {src}")
    rng = random.Random(seed)
    rng.shuffle(images)
    test_n = min(max_test, max(1, len(images) // 5))
    test_imgs = images[:test_n]
    train_imgs = images[test_n : test_n + max_train]

    out = OUT_ROOT / name
    for img in train_imgs:
        _copy(img, out / "train" / "good", "good_")
    for img in test_imgs:
        _copy(img, out / "test" / "good", "good_")

    return {"name": name, "train_good": len(train_imgs), "test_good": len(test_imgs),
            "test_defect": 0, "note": "plain folder subset"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert extra datasets to good-only layout")
    parser.add_argument("--pills", default="", help="YOLO medical-pills root")
    parser.add_argument("--bottle", default="", help="YOLO plastic-bottle root")
    parser.add_argument("--folder", action="append", default=[],
                        help="name=/path/to/image/folder (repeatable)")
    parser.add_argument("--max-train", type=int, default=300)
    parser.add_argument("--max-test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    if args.pills:
        records.append(convert_yolo("pills", Path(args.pills).expanduser(),
                                    args.max_train, args.max_test, args.seed))
    if args.bottle:
        records.append(convert_yolo("bottle", Path(args.bottle).expanduser(),
                                    args.max_train, args.max_test, args.seed))
    for spec in args.folder:
        if "=" not in spec:
            raise SystemExit(f"--folder expects name=path, got: {spec}")
        name, path = spec.split("=", 1)
        records.append(convert_folder(name, Path(path).expanduser(),
                                      args.max_train, args.max_test, args.seed))

    if not records:
        parser.print_help()
        return 2

    print("\nconverted:")
    for r in records:
        print(f"  {r['name']:<12} train_good={r['train_good']:<4} "
              f"test_good={r['test_good']:<3} test_defect={r['test_defect']:<3} ({r['note']})")
    print(f"\noutput root: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
