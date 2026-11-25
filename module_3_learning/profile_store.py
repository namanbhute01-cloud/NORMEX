"""M3: versioned, local, offline product-profile store.

Layout on disk::

    profiles/<product_id>/
    ├── current                 # "v003"
    └── v003/
        ├── meta.json
        ├── mean.npy
        ├── inv_cov.npy
        ├── channel_idx.npy
        ├── calib.json
        ├── preprocess.json
        └── train_report.json

Saves are atomic: files are written to a temporary directory inside the
product folder and renamed into place; ``current`` is updated last.  A
crash never leaves a half-written version active.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from module_0_common.types import Calibration
from module_3_learning.registry import get_algo

PROFILE_FORMAT = 1
PRODUCT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
VERSION_RE = re.compile(r"^v(\d{3,})$")


class ProfileError(Exception):
    """Base class for profile-store failures."""


class ProfileNotFoundError(ProfileError):
    """Requested product or version does not exist."""


class ProfileMismatchError(ProfileError):
    """Profile was trained with a different backbone/config than the running one."""


@dataclass
class ProfileData:
    """Everything needed to persist one trained model."""

    algo: Any
    channel_idx: torch.Tensor
    calib: Calibration
    preprocess: dict
    meta: dict
    train_report: dict = field(default_factory=dict)


@dataclass
class Profile:
    """A loaded, ready-to-use product profile."""

    product_id: str
    version: str
    path: Path
    meta: dict
    algo: Any
    channel_idx: torch.Tensor
    calib: Calibration
    preprocess: dict

    @property
    def input_size(self) -> int:
        return int(self.preprocess["size"])

    @property
    def pool(self) -> int:
        return int(self.meta["pool"])

    @property
    def d(self) -> int:
        return int(self.meta["d"])

    @property
    def hw(self) -> tuple[int, int]:
        hw = self.meta["hw"]
        return int(hw[0]), int(hw[1])


def _validate_product_id(product_id: str) -> None:
    if not PRODUCT_ID_RE.match(product_id or ""):
        raise ValueError(
            f"invalid product_id {product_id!r}: use 1-64 chars of A-Z, a-z, 0-9, '_' or '-'"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_profile_dir(directory: Path, data: ProfileData, product_id: str, version: str) -> None:
    """Write all profile files into an existing directory."""

    directory.mkdir(parents=True, exist_ok=True)
    state = data.algo.state()
    np.save(directory / "mean.npy", state["mean"])
    np.save(directory / "inv_cov.npy", state["inv_cov"])
    np.save(directory / "channel_idx.npy", data.channel_idx.cpu().numpy().astype(np.int64))

    meta = dict(data.meta)
    meta.update(
        {
            "format": PROFILE_FORMAT,
            "product_id": product_id,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    _write_json(directory / "meta.json", meta)
    _write_json(directory / "calib.json", data.calib.to_dict())
    _write_json(directory / "preprocess.json", data.preprocess)
    _write_json(directory / "train_report.json", data.train_report)


class ProfileStore:
    """Read/write product profiles under ``root`` (default ``profiles/``)."""

    def __init__(self, root: str | Path = "profiles") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ save
    def save(self, product_id: str, data: ProfileData) -> str:
        """Write a new version atomically and make it current.  Returns ``vNNN``."""

        _validate_product_id(product_id)
        product_dir = self.root / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        versions = self.list_versions(product_id)
        next_number = max((int(VERSION_RE.match(v).group(1)) for v in versions), default=0) + 1
        version = f"v{next_number:03d}"

        tmp_dir = product_dir / f".tmp_{version}_{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        final_dir = product_dir / version
        if final_dir.exists():
            raise ProfileError(f"profile version already exists: {final_dir}")

        try:
            _write_profile_dir(tmp_dir, data, product_id, version)
            os.replace(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        _atomic_write_text(product_dir / "current", version)
        return version

    # ------------------------------------------------------------------ load
    def load(
        self,
        product_id: str,
        version: Optional[str] = None,
        *,
        expected_config_hash: str,
        expected_backbone_sha256: str,
    ) -> Profile:
        """Load a profile, refusing silently-incompatible models."""

        _validate_product_id(product_id)
        product_dir = self.root / product_id
        if not product_dir.is_dir():
            raise ProfileNotFoundError(f"no profile for product {product_id!r} under {self.root}")

        if version is None:
            current = product_dir / "current"
            if not current.is_file():
                raise ProfileNotFoundError(f"profile {product_id!r} has no 'current' pointer")
            version = current.read_text(encoding="utf-8").strip()

        version_dir = product_dir / version
        if not version_dir.is_dir():
            raise ProfileNotFoundError(f"profile version not found: {version_dir}")

        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != PROFILE_FORMAT:
            raise ProfileError(
                f"unsupported profile format {meta.get('format')!r} "
                f"(this build expects {PROFILE_FORMAT}); migrate the profile"
            )

        if meta.get("config_hash") != expected_config_hash:
            raise ProfileMismatchError(
                f"config mismatch for {product_id}/{version}: profile was trained with "
                f"config_hash={meta.get('config_hash')!r}, running config is "
                f"{expected_config_hash!r}. Retrain or restore the matching config."
            )
        if meta.get("backbone_sha256") != expected_backbone_sha256:
            raise ProfileMismatchError(
                f"backbone mismatch for {product_id}/{version}: profile was trained with "
                f"weights sha256={meta.get('backbone_sha256')!r}, running weights are "
                f"{expected_backbone_sha256!r}."
            )

        mean = np.load(version_dir / "mean.npy")
        inv_cov = np.load(version_dir / "inv_cov.npy")
        channel_idx = torch.from_numpy(np.load(version_dir / "channel_idx.npy")).to(torch.int64)
        calib = Calibration.from_dict(json.loads((version_dir / "calib.json").read_text("utf-8")))
        preprocess = json.loads((version_dir / "preprocess.json").read_text("utf-8"))

        algo = get_algo(meta["algo"], {meta["algo"]: {"eps": float(meta["eps"])}})
        algo = algo.from_state(
            {"mean": mean, "inv_cov": inv_cov},
            eps=float(meta["eps"]),
            hw=(int(meta["hw"][0]), int(meta["hw"][1])),
        )

        return Profile(
            product_id=product_id,
            version=version,
            path=version_dir,
            meta=meta,
            algo=algo,
            channel_idx=channel_idx,
            calib=calib,
            preprocess=preprocess,
        )

    # -------------------------------------------------------------- registry
    def list_versions(self, product_id: str) -> list[str]:
        _validate_product_id(product_id)
        product_dir = self.root / product_id
        if not product_dir.is_dir():
            return []
        versions = [
            entry.name
            for entry in product_dir.iterdir()
            if entry.is_dir() and VERSION_RE.match(entry.name)
        ]
        return sorted(versions)

    def current_version(self, product_id: str) -> Optional[str]:
        _validate_product_id(product_id)
        current = self.root / product_id / "current"
        if not current.is_file():
            return None
        return current.read_text(encoding="utf-8").strip()

    def list_products(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        products = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not PRODUCT_ID_RE.match(entry.name):
                continue
            versions = self.list_versions(entry.name)
            if not versions:
                continue
            products.append(
                {
                    "product_id": entry.name,
                    "current": self.current_version(entry.name),
                    "versions": versions,
                }
            )
        return products

    def update_train_report(self, product_id: str, version: str, report: dict) -> None:
        """Rewrite a version's ``train_report.json`` after the save timing is known."""

        _validate_product_id(product_id)
        path = self.root / product_id / version / "train_report.json"
        if not path.is_file():
            raise ProfileNotFoundError(f"train report not found: {path}")
        _write_json(path, report)

    def rollback(self, product_id: str) -> str:
        """Repoint ``current`` to the previous version and return it."""

        _validate_product_id(product_id)
        versions = self.list_versions(product_id)
        if len(versions) < 2:
            raise ProfileError(f"nothing to roll back for {product_id!r}: only {versions}")
        current = self.current_version(product_id)
        if current not in versions:
            raise ProfileError(f"current version {current!r} is missing on disk")
        index = versions.index(current)
        if index == 0:
            raise ProfileError(f"no earlier version to roll back to for {product_id!r}")
        previous = versions[index - 1]
        _atomic_write_text(self.root / product_id / "current", previous)
        return previous
