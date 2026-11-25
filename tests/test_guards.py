"""Guards: no defect leakage, offline operation, determinism, no CUDA."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

from module_0_common.config import apply_overrides, config_hash
from module_3_learning.profile_store import ProfileStore
from module_3_learning.trainer import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_learning_modules_never_import_eval() -> None:
    for path in (PROJECT_ROOT / "module_3_learning").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "module_6_evaluation" not in text, f"{path} imports the evaluation module"
        assert "module_6_evaluation.mvtec" not in text, f"{path} imports the MVTec adapter"


def test_modules_have_no_cuda_references() -> None:
    for module_dir in PROJECT_ROOT.glob("module_*"):
        for path in module_dir.rglob("*.py"):
            lowered = path.read_text(encoding="utf-8").lower()
            assert ".cuda(" not in lowered, f"{path} references .cuda()"
            assert 'device="cuda"' not in lowered, f"{path} references a CUDA device"
            assert "cuda:" not in lowered, f"{path} references a CUDA device"


@pytest.mark.parametrize("bad_part", ["test", "ground_truth"])
def test_training_refuses_test_like_paths(tmp_path, base_cfg, bad_part) -> None:
    directory = tmp_path / bad_part / "good"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="test/defect"):
        train("guard", directory, base_cfg, backbone=None)


def _small_cfg(base_cfg: dict) -> dict:
    return apply_overrides(
        base_cfg,
        {
            "input.size": 128,
            "train.augment_factor": 0,
            "calibration.kfold": 2,
            "train.batch_size": 8,
        },
    )


def test_offline_train_and_inspect(ten_good, base_cfg, backbone, tmp_path, monkeypatch) -> None:
    def blocked(*args, **kwargs):
        raise RuntimeError("network access is forbidden at runtime")

    monkeypatch.setattr(socket.socket, "connect", blocked)

    cfg = _small_cfg(base_cfg)
    store = ProfileStore(tmp_path / "profiles")
    result = train("offline", ten_good, cfg, backbone, store)
    profile = store.load(
        "offline",
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )

    from module_1_acquisition.loader import list_images, load_frame
    from module_2_preprocessing.pipeline import preprocess
    from module_4_inspection.inspector import Inspector

    inspector = Inspector(backbone, profile, cfg)
    frame = load_frame(list_images(ten_good)[0])
    prepared = preprocess(
        frame,
        cfg,
        stored=profile.preprocess,
        quality_limits=profile.preprocess.get("quality_limits"),
    )
    inspection = inspector.inspect(prepared)

    assert result.version == "v001"
    assert np.isfinite(inspection.score)
    assert inspection.anomaly_map.shape == (128, 128)


def test_training_is_deterministic(ten_good, base_cfg, backbone, tmp_path) -> None:
    cfg = _small_cfg(base_cfg)
    store = ProfileStore(tmp_path / "profiles")

    train("det", ten_good, cfg, backbone, store)
    train("det", ten_good, cfg, backbone, store)

    first = store.load("det", "v001", expected_config_hash=config_hash(cfg),
                       expected_backbone_sha256=backbone.sha256)
    second = store.load("det", "v002", expected_config_hash=config_hash(cfg),
                        expected_backbone_sha256=backbone.sha256)

    assert np.allclose(np.load(first.path / "mean.npy"), np.load(second.path / "mean.npy"), atol=1e-6)
    assert np.allclose(np.load(first.path / "inv_cov.npy"),
                       np.load(second.path / "inv_cov.npy"), atol=1e-6)
    assert first.calib.t_lo == pytest.approx(second.calib.t_lo)
    assert first.calib.t_hi == pytest.approx(second.calib.t_hi)
