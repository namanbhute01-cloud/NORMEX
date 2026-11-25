"""Synthetic end-to-end: train on good only, inspect good + defects.

This is the gate that must pass before any MVTec run.  It uses obvious
synthetic defects, so it tests the pipeline, not the state of the art.
"""

from __future__ import annotations

import cv2
import numpy as np

from module_0_common.config import apply_overrides, config_hash
from module_6_evaluation.metrics import image_auroc
from module_1_acquisition.loader import list_images, load_frame
from module_2_preprocessing.pipeline import preprocess
from module_3_learning.profile_store import ProfileStore
from module_3_learning.trainer import train
from module_4_inspection.anomaly_map import overlay
from module_4_inspection.inspector import Inspector
from module_5_decision.decide import decide


def _inspect_paths(paths, inspector, profile, cfg):
    results = []
    for path in paths:
        prepared = preprocess(
            load_frame(path),
            cfg,
            stored=profile.preprocess,
            quality_limits=profile.preprocess.get("quality_limits"),
        )
        inspection = inspector.inspect(prepared) if prepared.quality_ok else None
        verdict = decide(
            prepared,
            inspection,
            profile.calib,
            "synth",
            profile.version,
            inspection.latency_ms if inspection else 0.0,
        )
        results.append((path, prepared, inspection, verdict))
    return results


def test_train_inspect_end_to_end(synthetic_dataset, base_cfg, backbone, tmp_path) -> None:
    cfg = apply_overrides(
        base_cfg,
        {"train.augment_factor": 2, "calibration.kfold": 3},
    )
    store = ProfileStore(tmp_path / "profiles")
    result = train("synth", synthetic_dataset["good"], cfg, backbone, store)

    # Spec §8.2 dev-loop gate.  This is wall-clock and assumes an ordinary
    # laptop; the measured run on this 2-core dev box takes ~43 s.
    assert result.total_seconds < 120.0, "synthetic training must stay well under 2 minutes"
    profile = store.load(
        "synth",
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )
    inspector = Inspector(backbone, profile, cfg)

    good = _inspect_paths(list_images(synthetic_dataset["new_good"]), inspector, profile, cfg)
    defects = _inspect_paths(list_images(synthetic_dataset["defect"]), inspector, profile, cfg)

    labels = [0] * len(good) + [1] * len(defects)
    scores = [verdict.score for _, _, _, verdict in good + defects]
    auroc = image_auroc(labels, scores)
    assert auroc >= 0.95, f"image AUROC too low: {auroc:.4f}"

    good_scores = np.array([verdict.score for _, _, _, verdict in good])
    defect_scores = np.array([verdict.score for _, _, _, verdict in defects])
    assert defect_scores.mean() >= 1.5 * good_scores.mean(), (
        f"defect scores not clearly higher: good={good_scores.mean():.3f} "
        f"defect={defect_scores.mean():.3f}"
    )

    flagged = np.mean([verdict.status in {"DEFECTIVE", "REVIEW"} for _, _, _, verdict in defects])
    assert flagged >= 0.80, f"only {flagged:.0%} of defects were flagged"

    false_defective = np.mean([verdict.status == "DEFECTIVE" for _, _, _, verdict in good])
    assert false_defective <= 0.20, f"{false_defective:.0%} of good images marked DEFECTIVE"

    print(
        f"[synthetic] size={profile.input_size} AUROC={auroc:.4f} "
        f"good_mean={good_scores.mean():.2f} defect_mean={defect_scores.mean():.2f} "
        f"flagged={flagged:.0%} false_defective={false_defective:.0%} "
        f"train={result.total_seconds:.1f}s"
    )

    # Localization sanity for the two surface defects.
    for path, _, inspection, _ in defects:
        kind = path.stem.split("_")[-1]
        if kind not in {"scratch", "stain"}:
            continue
        mask_path = synthetic_dataset["masks"] / f"{path.stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        peak_y, peak_x = np.unravel_index(np.argmax(inspection.anomaly_map), inspection.anomaly_map.shape)
        ys, xs = np.nonzero(mask)
        distance = np.sqrt((ys - peak_y) ** 2 + (xs - peak_x) ** 2).min()
        assert distance <= 24.0, f"{path.name}: peak {distance:.1f}px from the defect"

    # Heatmap is a valid PNG at the input resolution.
    _, prepared, inspection, _ = defects[0]
    heatmap = overlay(prepared.display, inspection.anomaly_map, 0.0, profile.calib.t_hi * 1.5)
    heatmap_path = tmp_path / "heatmap.png"
    assert cv2.imwrite(str(heatmap_path), heatmap)
    written = cv2.imread(str(heatmap_path))
    assert written is not None and written.shape == (profile.input_size, profile.input_size, 3)


def test_retrain_bumps_version_and_rollback(ten_good, base_cfg, backbone, tmp_path) -> None:
    cfg = apply_overrides(
        base_cfg,
        {
            "input.size": 128,
            "train.augment_factor": 0,
            "calibration.kfold": 2,
            "train.batch_size": 8,
        },
    )
    store = ProfileStore(tmp_path / "profiles")

    first = train("versioned", ten_good, cfg, backbone, store)
    second = train("versioned", ten_good, cfg, backbone, store)
    assert (first.version, second.version) == ("v001", "v002")
    assert store.current_version("versioned") == "v002"

    assert store.rollback("versioned") == "v001"
    profile = store.load(
        "versioned",
        expected_config_hash=config_hash(cfg),
        expected_backbone_sha256=backbone.sha256,
    )
    assert profile.version == "v001"
