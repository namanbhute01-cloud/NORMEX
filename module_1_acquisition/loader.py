"""M1 (minimal): read photographs from disk.

No camera in this phase.  The loader is deliberately dumb: it returns a
BGR ``Frame`` and never touches labels or masks.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from module_0_common.types import Frame

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(path: str | Path) -> list[Path]:
    """Return sorted image paths from a file or a directory.

    A missing path or an empty directory raises ``FileNotFoundError`` so the
    CLI fails loudly instead of silently training on nothing.
    """

    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"image path not found: {path}")

    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in: {path}")
    return images


def load_frame(path: str | Path) -> Frame:
    """Read one image as a 3-channel BGR ``Frame``."""

    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image (corrupt or unsupported): {path}")
    return Frame(image=image, source=str(path))
