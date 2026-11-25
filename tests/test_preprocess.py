"""T2/T8: preprocessing, quality gate and defect-safe augmentation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from module_0_common.config import apply_overrides
from module_0_common.types import Frame
from module_1_acquisition.loader import list_images, load_frame
from module_2_preprocessing import align, augment, denoise, illumination, pipeline, quality, transform
from module_2_preprocessing.transform import prepare


def test_prepare_shapes_and_normalization(synthetic_dataset) -> None:
    frame = load_frame(list_images(synthetic_dataset["good"])[0])
    prepared = prepare(frame, size=256)

    assert tuple(prepared.tensor.shape) == (1, 3, 256, 256)
    assert prepared.tensor.dtype == torch.float32
    assert prepared.display.shape == (256, 256, 3)
    assert prepared.display.dtype == np.uint8
    assert prepared.quality_ok is True
    assert float(prepared.tensor.min()) > -6.0
    assert float(prepared.tensor.max()) < 6.0


def test_quality_gate_rejects_blur_and_darkness(synthetic_dataset) -> None:
    frames = [load_frame(p) for p in list_images(synthetic_dataset["good"])[:10]]
    limits = quality.learn_limits([quality.measure(frame.image) for frame in frames])

    original = frames[0].image
    ok, message = quality.check(quality.measure(original), limits)
    assert ok, message

    blurred = cv2.GaussianBlur(original, (0, 0), sigmaX=15)
    assert not quality.check(quality.measure(blurred), limits)[0]

    dark = (original.astype(np.float32) * 0.3).astype(np.uint8)
    assert not quality.check(quality.measure(dark), limits)[0]


def test_prepare_marks_bad_image(synthetic_dataset) -> None:
    frame = load_frame(list_images(synthetic_dataset["good"])[0])
    frame.image = cv2.GaussianBlur(frame.image, (0, 0), sigmaX=15)
    limits = {"blur_min": 1000.0, "mean_min": 0.0, "mean_max": 255.0, "sat_max": 1.0}
    prepared = prepare(frame, 256, {"enabled": True}, limits)
    assert prepared.quality_ok is False
    assert "blurry" in prepared.quality_msg


def test_augment_is_mild_deterministic_and_changes_image(synthetic_dataset) -> None:
    frame = load_frame(list_images(synthetic_dataset["good"])[0])
    cfg = {"shift": 0.03, "rot_deg": 3.0, "brightness": 0.10, "contrast": 0.10, "flip": False}

    first = augment.augment(frame.image, np.random.default_rng(7), cfg)
    second = augment.augment(frame.image, np.random.default_rng(7), cfg)

    assert np.array_equal(first, second)
    assert first.shape == frame.image.shape
    assert first.dtype == np.uint8
    assert not np.array_equal(first, frame.image)
    change = np.mean(np.abs(first.astype(np.int16) - frame.image.astype(np.int16)))
    assert change < 25.0


def test_augment_flip_only_when_enabled(synthetic_dataset) -> None:
    frame = load_frame(list_images(synthetic_dataset["good"])[0])
    cfg_flip = {"shift": 0.0, "rot_deg": 0.0, "brightness": 0.0, "contrast": 0.0, "flip": True}
    # rng(2).random() < 0.5, so this call flips.
    flipped = augment.augment(frame.image, np.random.default_rng(2), cfg_flip)
    assert np.array_equal(flipped, cv2.flip(frame.image, 1))

    cfg_no_flip = dict(cfg_flip, flip=False)
    unchanged = augment.augment(frame.image, np.random.default_rng(2), cfg_no_flip)
    assert np.array_equal(unchanged, frame.image)


def test_prepare_preserves_rgb_channel_order(base_cfg) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 2] = 255  # OpenCV BGR: pure red
    prepared = pipeline.preprocess(Frame(image=image, source="rgb"), base_cfg)

    rgb = prepared.tensor[0].numpy().transpose(1, 2, 0)
    rgb = rgb * transform.IMAGENET_STD + transform.IMAGENET_MEAN
    assert rgb[32, 32, 0] > 0.95
    assert rgb[32, 32, 1] < 0.05
    assert rgb[32, 32, 2] < 0.05


def test_letterbox_preserves_aspect_ratio(base_cfg) -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    prepared = pipeline.preprocess(Frame(image=image, source="wide"), base_cfg)
    geometry = prepared.geometry

    assert (geometry["new_w"], geometry["new_h"]) == (256, 128)
    assert (geometry["pad_x"], geometry["pad_y"]) == (0, 64)
    assert geometry["new_w"] / geometry["src_w"] == pytest.approx(
        geometry["new_h"] / geometry["src_h"]
    )
    assert np.all(prepared.display[:64] == 0)
    assert np.all(prepared.display[-64:] == 0)


def test_roi_is_cropped_and_recorded(base_cfg) -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:, :200] = (255, 0, 0)   # BGR blue, outside ROI
    image[:, 200:] = (0, 0, 255)   # BGR red, inside ROI
    cfg = apply_overrides(
        base_cfg,
        {
            "preprocess.roi.enabled": True,
            "preprocess.roi.x": 200,
            "preprocess.roi.y": 0,
            "preprocess.roi.width": 200,
            "preprocess.roi.height": 300,
        },
    )
    prepared = pipeline.preprocess(Frame(image=image, source="roi"), cfg)

    assert prepared.geometry["roi"] == [200, 0, 200, 300]
    rgb = prepared.tensor[0].numpy().transpose(1, 2, 0)
    rgb = rgb * transform.IMAGENET_STD + transform.IMAGENET_MEAN
    center = rgb[128, 128]
    assert center[0] > 0.95 and center[2] < 0.05


def test_map_to_source_lands_on_original_coordinates(base_cfg) -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    prepared = pipeline.preprocess(Frame(image=image, source="map"), base_cfg)

    anomaly = np.zeros((256, 256), dtype=np.float32)
    anomaly[128, 128] = 10.0
    source = transform.map_to_source(anomaly, prepared.geometry)
    peak_y, peak_x = np.unravel_index(np.argmax(source), source.shape)

    assert source.shape == (200, 400)
    assert np.hypot(peak_x - 200, peak_y - 100) <= 6.0


def test_alignment_reduces_known_shift() -> None:
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(
        (rng.normal(0, 1, (128, 128)) * 40 + 128).astype(np.uint8), (0, 0), 2
    )
    reference = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    matrix = np.float32([[1, 0, 5], [0, 1, -3]])
    shifted = cv2.warpAffine(
        reference, matrix, (128, 128), borderMode=cv2.BORDER_REFLECT_101
    )
    cfg = {"max_shift_px": 20.0, "max_rotation_deg": 5.0}

    aligned, info = align.align(shifted, reference, cfg)
    before = np.abs(shifted.astype(np.float32) - reference.astype(np.float32)).mean()
    after = np.abs(aligned.astype(np.float32) - reference.astype(np.float32)).mean()

    assert info["ok"] and info["applied"]
    assert after < 0.5 * before


def test_alignment_refuses_large_correction() -> None:
    rng = np.random.default_rng(6)
    base = cv2.GaussianBlur(
        (rng.normal(0, 1, (128, 128)) * 40 + 128).astype(np.uint8), (0, 0), 2
    )
    reference = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    shifted = cv2.warpAffine(
        reference, np.float32([[1, 0, 20], [0, 1, 0]]), (128, 128),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    info = align.estimate(reference, shifted, {"max_shift_px": 5.0, "max_rotation_deg": 2.0})
    assert not info["ok"]
    assert "too large" in info["message"]


def test_illumination_background_preserves_a_scratch() -> None:
    yy, xx = np.mgrid[0:128, 0:128]
    background = (90.0 + 70.0 * xx / 127.0).astype(np.float32)
    image = np.dstack([background] * 3).astype(np.uint8)
    image[:, 60:63] = (20, 20, 20)

    corrected = illumination.apply(
        image, {"enabled": True, "method": "background", "sigma": 20.0}
    )

    def contrast(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(abs(gray[:, 60:63].mean() - gray[:, 50:55].mean()))

    assert contrast(corrected) > 0.7 * contrast(image)


def test_denoise_is_off_by_default_and_opt_in() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert denoise.apply(image, {"enabled": False}) is image

    noisy = image.copy()
    noisy[::2, ::2] = 200
    cleaned = denoise.apply(noisy, {"enabled": True, "method": "median", "ksize": 3})
    assert cleaned.shape == noisy.shape and cleaned.dtype == np.uint8
    assert not np.array_equal(cleaned, noisy)


def test_preprocess_records_latency_and_uses_stored_geometry(base_cfg) -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    frame = Frame(image=image, source="stored")
    prepared = pipeline.preprocess(frame, base_cfg)
    assert prepared.latency_ms > 0
    assert {"roi", "quality", "tensor"} <= set(prepared.stages_ms)

    stored = pipeline.default_preprocess_cfg()
    stored.update({"size": 128, "quality_enabled": False})
    prepared_small = pipeline.preprocess(frame, base_cfg, stored=stored, apply_quality=False)
    assert tuple(prepared_small.tensor.shape) == (1, 3, 128, 128)


def test_quality_rejects_wrong_size_and_missing_part(synthetic_dataset, base_cfg) -> None:
    frames = [load_frame(p) for p in list_images(synthetic_dataset["good"])[:10]]
    measures = [pipeline.measure_quality(frame, base_cfg) for frame in frames]
    limits = quality.learn_limits(measures)

    wrong_size = cv2.resize(frames[0].image, (128, 128))
    ok, message = quality.check(quality.measure(wrong_size), limits)
    assert not ok and "wrong image size" in message

    blank = np.full_like(frames[0].image, 90)
    ok, message = quality.check(quality.measure(blank), limits)
    assert not ok and "foreground" in message


def test_aligned_heatmap_maps_back_to_the_original_pose() -> None:
    rng = np.random.default_rng(8)
    base = cv2.GaussianBlur(
        (rng.normal(0, 1, (128, 128)) * 40 + 128).astype(np.uint8), (0, 0), 2
    )
    reference = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    shifted = cv2.warpAffine(
        reference, np.float32([[1, 0, 5], [0, 1, -3]]), (128, 128),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    info = align.estimate(
        reference, shifted, {"max_shift_px": 20.0, "max_rotation_deg": 5.0}
    )
    assert info["ok"]

    geometry = {
        "src_w": 128,
        "src_h": 128,
        "size": 128,
        "keep_aspect": True,
        "new_w": 128,
        "new_h": 128,
        "pad_x": 0,
        "pad_y": 0,
        "roi": [0, 0, 128, 128],
        "align": info,
    }
    anomaly = np.zeros((128, 128), dtype=np.float32)
    anomaly[64, 64] = 10.0  # marker in the aligned/reference frame
    source = transform.map_to_source(anomaly, geometry)
    peak_y, peak_x = np.unravel_index(np.argmax(source), source.shape)

    assert np.hypot(peak_x - 69, peak_y - 61) <= 2.0


def test_heatmap_mask_keeps_outside_roi_untouched() -> None:
    from module_4_inspection.anomaly_map import overlay

    display = np.full((10, 10, 3), 100, dtype=np.uint8)
    anomaly = np.ones((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255

    heatmap = overlay(display, anomaly, 0.0, 1.0, 0.5, mask)
    assert np.array_equal(heatmap[0, 0], display[0, 0])
    assert not np.array_equal(heatmap[5, 5], display[5, 5])


def test_small_scratch_survives_resize(base_cfg) -> None:
    image = np.full((512, 512, 3), 180, dtype=np.uint8)
    cv2.line(image, (100, 256), (400, 256), (20, 20, 20), 4)
    prepared = pipeline.preprocess(Frame(image=image, source="scratch"), base_cfg)

    line = prepared.display[128, 50:200, 0].astype(np.float32)
    background = prepared.display[120:126, 50:200, 0].astype(np.float32)
    assert line.min() < 110
    assert background.mean() - line.mean() > 30.0
