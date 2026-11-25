"""T9: profile save/load, versioning, rollback and mismatch guards."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from module_0_common.types import Calibration
from module_3_learning import profile_store
from module_3_learning.embedding import make_channel_idx
from module_3_learning.padim import PaDiM
from module_3_learning.profile_store import (
    ProfileData,
    ProfileMismatchError,
    ProfileStore,
)


def _data(config_hash: str = "cfg123", backbone_hash: str = "bb456") -> ProfileData:
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(20, 8, 8, 8, generator=generator)
    algo = PaDiM(eps=0.01).fit(embeddings)
    calib = Calibration(
        t_lo=1.0, t_hi=1.15, good_scores=[0.9, 1.0], mu=0.95, sd=0.05,
        sigma_k=3.0, review_margin=1.15, kfold=2,
    )
    meta = {
        "algo": "padim", "eps": 0.01, "d": 8, "pool": 2, "hw": [8, 8],
        "config_hash": config_hash, "backbone_sha256": backbone_hash,
        "input_size": 256, "n_good": 20, "n_aug": 0, "kfold": 2,
    }
    preprocess = {
        "size": 256,
        "quality_enabled": True,
        "quality_limits": {"blur_min": 1.0, "mean_min": 1.0, "mean_max": 254.0, "sat_max": 0.05},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    return ProfileData(
        algo=algo,
        channel_idx=make_channel_idx(448, 8, seed=0),
        calib=calib,
        preprocess=preprocess,
        meta=meta,
        train_report={"stages_ms": {}},
    )


def _score(profile, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return profile.algo.score(torch.randn(2, 8, 8, 8, generator=generator))


def test_save_load_round_trip(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    version = store.save("part_a", _data())
    assert version == "v001"
    assert (tmp_path / "part_a" / "current").read_text().strip() == "v001"

    profile = store.load(
        "part_a", expected_config_hash="cfg123", expected_backbone_sha256="bb456"
    )
    assert profile.version == "v001"
    assert torch.equal(profile.channel_idx, make_channel_idx(448, 8, seed=0))
    assert _score(profile).shape == (2, 8, 8)

    saved = np.load(profile.path / "mean.npy")
    assert saved.dtype == np.float32


def test_two_saves_and_rollback(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    store.save("part_a", _data())
    second = _data()
    second.algo = PaDiM(eps=0.01).fit(torch.randn(20, 8, 8, 8, generator=torch.Generator().manual_seed(9)))
    version_two = store.save("part_a", second)

    assert version_two == "v002"
    assert store.current_version("part_a") == "v002"
    scores_two = _score(store.load("part_a", expected_config_hash="cfg123",
                                   expected_backbone_sha256="bb456"))

    assert store.rollback("part_a") == "v001"
    assert store.current_version("part_a") == "v001"
    scores_one = _score(store.load("part_a", expected_config_hash="cfg123",
                                   expected_backbone_sha256="bb456"))
    assert not torch.allclose(scores_one, scores_two)


def test_mismatches_raise(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    store.save("part_a", _data())

    with pytest.raises(ProfileMismatchError, match="config mismatch"):
        store.load("part_a", expected_config_hash="other", expected_backbone_sha256="bb456")

    with pytest.raises(ProfileMismatchError, match="backbone mismatch"):
        store.load("part_a", expected_config_hash="cfg123", expected_backbone_sha256="other")


def test_bad_product_id_rejected(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    with pytest.raises(ValueError, match="invalid product_id"):
        store.save("../escape", _data())


def test_missing_profile_raises(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    with pytest.raises(profile_store.ProfileNotFoundError):
        store.load("nope", expected_config_hash="cfg123", expected_backbone_sha256="bb456")


def test_interrupted_save_keeps_current(tmp_path, monkeypatch) -> None:
    store = ProfileStore(tmp_path)
    store.save("part_a", _data())
    assert store.current_version("part_a") == "v001"

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profile_store, "_write_profile_dir", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        store.save("part_a", _data())

    assert store.current_version("part_a") == "v001"
    assert store.list_versions("part_a") == ["v001"]
    assert not any(p.name.startswith(".tmp") for p in (tmp_path / "part_a").iterdir())


def test_list_products(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    store.save("part_a", _data())
    store.save("part_b", _data())
    products = {p["product_id"]: p for p in store.list_products()}
    assert set(products) == {"part_a", "part_b"}
    assert products["part_a"]["current"] == "v001"
