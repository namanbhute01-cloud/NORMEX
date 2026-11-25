#!/usr/bin/env python3
"""One-time, online download of ImageNet ResNet18 weights.

This is the ONLY file in the repository allowed to use the network.
Everything under ``module_*/`` loads weights from disk.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
EXPECTED_PREFIX = "f37072fd"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "pretrained" / "resnet18.pth"
HASH_FILE = PROJECT_ROOT / "pretrained" / "resnet18.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.is_file():
        digest = sha256_file(TARGET)
        if digest.startswith(EXPECTED_PREFIX):
            HASH_FILE.write_text(digest + "\n", encoding="utf-8")
            print(f"weights already present and verified: {TARGET}")
            print(f"sha256: {digest}")
            return
        print(f"existing weights have unexpected sha256 ({digest}); re-downloading")

    partial = TARGET.with_suffix(".pth.part")
    digest = hashlib.sha256()
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            seen += len(chunk)
            if total:
                pct = 100.0 * seen / total
                print(f"\r  {seen / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="")
    print()

    actual = digest.hexdigest()
    if not actual.startswith(EXPECTED_PREFIX):
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"sha256 mismatch: expected prefix {EXPECTED_PREFIX!r}, got {actual!r}. "
            "Refusing to install unverified weights."
        )

    os.replace(partial, TARGET)
    HASH_FILE.write_text(actual + "\n", encoding="utf-8")
    print(f"installed {TARGET}")
    print(f"sha256: {actual}")


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # noqa: BLE001 - CLI entry point
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
