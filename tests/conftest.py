"""Shared fixtures: deterministic synthetic dataset, config, backbone.

The synthetic set is deliberately simple and controlled: a textured
background with a bright disc, small pose/brightness variation for the good
images, and three obvious defect types (scratch, stain, missing chunk).
It is a *sanity harness*, not a benchmark.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_0_common.config import load_config  # noqa: E402

SIZE = 256


def _base_image(rng: np.random.Generator) -> np.ndarray:
    low = rng.normal(0.0, 1.0, (8, 8)).astype(np.float32)
    low = cv2.resize(low, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(low, (0, 0), 6)
    low = (low - low.min()) / (low.max() - low.min() + 1e-6)
    background = 55.0 + 70.0 * low
    image = np.dstack([background] * 3)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cy = cx = SIZE // 2
    radius = 0.27 * SIZE
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = distance < radius
    shade = 205.0 - 30.0 * (distance / radius)
    for channel in range(3):
        image[..., channel][disc] = shade[disc]

    for fx, fy, fr in [(cx - 42, cy - 42, 9), (cx + 42, cy + 42, 9),
                       (cx - 42, cy + 42, 9), (cx + 42, cy - 42, 9)]:
        hole = (xx - fx) ** 2 + (yy - fy) ** 2 < fr * fr
        for channel in range(3):
            image[..., channel][hole] = 95.0

    return np.clip(image, 0, 255).astype(np.uint8)


def _good_image(rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    dx, dy = rng.uniform(-2.0, 2.0, size=2)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(base, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT_101)
    alpha = 1.0 + rng.uniform(-0.05, 0.05)
    beta = rng.uniform(-5.0, 5.0)
    return np.clip(shifted.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _defect_image(
    rng: np.random.Generator, base: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    image = _good_image(rng, base)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    if kind == "scratch":
        x1, y1 = int(rng.integers(70, 186)), int(rng.integers(95, 160))
        x2, y2 = int(rng.integers(70, 186)), int(rng.integers(95, 160))
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        image = image.copy()
        image[mask > 0] = (25, 25, 25)
    elif kind == "stain":
        cx, cy = int(rng.integers(95, 161)), int(rng.integers(95, 161))
        radius = int(rng.integers(9, 15))
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        image = image.copy()
        image[mask > 0] = (60, 55, 50)
    elif kind == "missing":
        # A bite taken out of the top edge of the disc.
        cx = int(rng.integers(115, 141))
        cy = 55
        radius = 20
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        image = image.copy()
        image[mask > 0] = (80, 80, 80)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown synthetic defect kind: {kind}")

    return image, mask


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(1234)
    base = _base_image(rng)

    good_dir = root / "good"
    defect_dir = root / "defect"
    mask_dir = root / "masks"
    new_good_dir = root / "new_good"
    for directory in (good_dir, defect_dir, mask_dir, new_good_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index in range(30):
        cv2.imwrite(str(good_dir / f"good_{index:03d}.png"), _good_image(rng, base))

    defect_types: list[str] = []
    for index in range(15):
        kind = ["scratch", "stain", "missing"][index % 3]
        image, mask = _defect_image(rng, base, kind)
        name = f"defect_{index:03d}_{kind}"
        cv2.imwrite(str(defect_dir / f"{name}.png"), image)
        cv2.imwrite(str(mask_dir / f"{name}_mask.png"), mask)
        defect_types.append(kind)

    for index in range(10):
        cv2.imwrite(str(new_good_dir / f"new_good_{index:03d}.png"), _good_image(rng, base))

    return {
        "root": root,
        "good": good_dir,
        "defect": defect_dir,
        "masks": mask_dir,
        "new_good": new_good_dir,
        "defect_types": defect_types,
    }


@pytest.fixture(scope="session")
def ten_good(tmp_path_factory: pytest.TempPathFactory, synthetic_dataset: dict) -> Path:
    """A directory with exactly 10 good images (minimum training set)."""

    directory = tmp_path_factory.mktemp("ten_good")
    for path in sorted(synthetic_dataset["good"].glob("*.png"))[:10]:
        shutil.copy(path, directory / path.name)
    return directory


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def weights_path() -> Path:
    path = PROJECT_ROOT / "pretrained" / "resnet18.pth"
    if not path.is_file():
        pytest.skip("pretrained/resnet18.pth missing; run scripts/fetch_weights.py once")
    return path


@pytest.fixture(scope="session")
def backbone(weights_path: Path):
    from module_3_learning.backbone import Backbone

    return Backbone(weights_path)
