"""Deterministic synthetic parts for demos and plumbing tests.

These images are a *sanity harness*, not a benchmark.  The generator is kept
here (M6) so scripts and tests can share it; production learning/inspection
modules never import it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def base_image(size: int = 256, seed: int = 1234) -> np.ndarray:
    """Textured background with a shaded disc and four holes."""

    rng = np.random.default_rng(seed)
    low = rng.normal(0.0, 1.0, (8, 8)).astype(np.float32)
    low = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(low, (0, 0), max(1.0, size / 42.0))
    low = (low - low.min()) / (low.max() - low.min() + 1e-6)
    background = 55.0 + 70.0 * low
    image = np.dstack([background] * 3)

    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size // 2
    radius = 0.27 * size
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = distance < radius
    shade = 205.0 - 30.0 * (distance / radius)
    for channel in range(3):
        image[..., channel][disc] = shade[disc]

    offset = int(round(42 * size / 256))
    hole_radius = max(2, int(round(9 * size / 256)))
    for fx, fy in [
        (cx - offset, cy - offset),
        (cx + offset, cy + offset),
        (cx - offset, cy + offset),
        (cx + offset, cy - offset),
    ]:
        hole = (xx - fx) ** 2 + (yy - fy) ** 2 < hole_radius * hole_radius
        for channel in range(3):
            image[..., channel][hole] = 95.0

    return np.clip(image, 0, 255).astype(np.uint8)


def good_image(rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    """A good part with small pose and brightness variation."""

    size = base.shape[0]
    dx, dy = rng.uniform(-2.0, 2.0, size=2) * (size / 256.0)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(base, matrix, (size, size), borderMode=cv2.BORDER_REFLECT_101)
    alpha = 1.0 + rng.uniform(-0.05, 0.05)
    beta = rng.uniform(-5.0, 5.0)
    return np.clip(shifted.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def defect_image(
    rng: np.random.Generator, base: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """A good part with one obvious synthetic defect and its mask."""

    size = base.shape[0]
    image = good_image(rng, base)
    mask = np.zeros((size, size), dtype=np.uint8)

    def px(value: float) -> int:
        return int(round(value * size))

    if kind == "scratch":
        x1, y1 = px(rng.uniform(0.27, 0.73)), px(rng.uniform(0.37, 0.63))
        x2, y2 = px(rng.uniform(0.27, 0.73)), px(rng.uniform(0.37, 0.63))
        cv2.line(mask, (x1, y1), (x2, y2), 255, max(2, px(3 / 256)))
        image = image.copy()
        image[mask > 0] = (25, 25, 25)
    elif kind == "stain":
        cx, cy = px(rng.uniform(0.37, 0.63)), px(rng.uniform(0.37, 0.63))
        radius = max(2, px(rng.uniform(0.035, 0.058)))
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        image = image.copy()
        image[mask > 0] = (60, 55, 50)
    elif kind == "missing":
        # A bite taken out of the top edge of the disc.
        cx = px(rng.uniform(0.45, 0.55))
        cy = px(0.215)
        radius = max(3, px(0.078))
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        image = image.copy()
        image[mask > 0] = (80, 80, 80)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown synthetic defect kind: {kind}")

    return image, mask


def generate_dataset(
    root: str | Path,
    *,
    n_good: int = 30,
    n_new_good: int = 10,
    n_defect: int = 15,
    size: int = 256,
    seed: int = 1234,
) -> dict:
    """Write ``good/``, ``new_good/``, ``defect/`` and ``masks/`` under root."""

    root = Path(root)
    rng = np.random.default_rng(seed)
    base = base_image(size=size, seed=seed)

    directories = {
        name: root / name for name in ("good", "new_good", "defect", "masks")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for index in range(int(n_good)):
        cv2.imwrite(str(directories["good"] / f"good_{index:03d}.png"), good_image(rng, base))
    for index in range(int(n_new_good)):
        cv2.imwrite(
            str(directories["new_good"] / f"new_good_{index:03d}.png"), good_image(rng, base)
        )

    kinds: list[str] = []
    for index in range(int(n_defect)):
        kind = ["scratch", "stain", "missing"][index % 3]
        image, mask = defect_image(rng, base, kind)
        name = f"defect_{index:03d}_{kind}"
        cv2.imwrite(str(directories["defect"] / f"{name}.png"), image)
        cv2.imwrite(str(directories["masks"] / f"{name}_mask.png"), mask)
        kinds.append(kind)

    return {
        "root": root,
        "size": int(size),
        "seed": int(seed),
        "good": directories["good"],
        "new_good": directories["new_good"],
        "defect": directories["defect"],
        "masks": directories["masks"],
        "n_good": int(n_good),
        "n_new_good": int(n_new_good),
        "n_defect": int(n_defect),
        "defect_types": kinds,
    }
