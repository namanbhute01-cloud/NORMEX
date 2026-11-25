"""MVTec AD adapter — the *only* module allowed to read test/ and ground_truth/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from module_1_acquisition.loader import IMAGE_EXTS


@dataclass
class TestImage:
    path: Path
    label: int  # 0 = good, 1 = defect
    defect_type: str
    mask_path: Optional[Path] = None


def category_root(dataset_root: str | Path, category: str) -> Path:
    root = Path(dataset_root) / category
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec category not found: {root}")
    return root


def list_categories(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec dataset root not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "train" / "good").is_dir())


def train_good_dir(dataset_root: str | Path, category: str) -> Path:
    path = category_root(dataset_root, category) / "train" / "good"
    if not path.is_dir():
        raise FileNotFoundError(f"missing train/good for {category}: {path}")
    return path


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def list_test(dataset_root: str | Path, category: str) -> list[TestImage]:
    """All test images with labels and (when available) ground-truth masks."""

    root = category_root(dataset_root, category)
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing test split for {category}: {test_dir}")

    items: list[TestImage] = []
    for type_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        defect_type = type_dir.name
        label = 0 if defect_type == "good" else 1
        for image_path in _images_in(type_dir):
            mask_path = root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            items.append(
                TestImage(
                    path=image_path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=mask_path if label == 1 and mask_path.is_file() else None,
                )
            )
    if not items:
        raise FileNotFoundError(f"no test images for {category} under {test_dir}")
    return items


def load_mask(mask_path: Path, size: int | None = None) -> np.ndarray:
    """Binary mask, optionally resized to ``size``x``size`` (nearest neighbour)."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read mask: {mask_path}")
    if size is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)
