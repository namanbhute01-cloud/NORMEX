"""T7: k-fold calibration and thresholds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from module_3_learning.padim import PaDiM
from module_4_inspection.anomaly_map import image_score, to_map
from module_5_decision.calibrate import (
    kfold_good_scores,
    kfold_indices,
    make_calibration,
)


def _features(n: int = 12, d: int = 8, h: int = 8, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, h, h, generator=generator)


def test_folds_partition_all_indices_once() -> None:
    n, k = 17, 5
    folds = kfold_indices(n, k, seed=0)
    assert len(folds) == k
    seen: list[int] = []
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == n
        seen.extend(val_idx.tolist())
    assert sorted(seen) == list(range(n))


def test_held_out_scores_length_and_thresholds() -> None:
    E = _features(n=12)
    scores = kfold_good_scores(
        E, None, None, lambda: PaDiM(eps=0.01), k=3, seed=0,
        size=64, sigma=2.0, mode="max", topk_frac=0.01,
    )
    assert scores.shape == (12,)
    assert np.all(np.isfinite(scores))

    calib = make_calibration(scores, sigma_k=3.0, review_margin=1.15, k=3)
    assert calib.t_lo >= float(scores.max())
    assert calib.t_hi == pytest.approx(calib.t_lo * 1.15)
    assert len(calib.good_scores) == 12


def test_too_few_good_images_raises() -> None:
    E = _features(n=9)
    with pytest.raises(ValueError, match="at least 10"):
        kfold_good_scores(
            E, None, None, lambda: PaDiM(eps=0.01), k=3, seed=0,
            size=64, sigma=2.0, mode="max", topk_frac=0.01,
        )


def test_held_out_scores_are_not_lower_than_in_sample() -> None:
    E = _features(n=15)
    held_out = kfold_good_scores(
        E, None, None, lambda: PaDiM(eps=0.01), k=3, seed=0,
        size=64, sigma=2.0, mode="max", topk_frac=0.01,
    )

    model = PaDiM(eps=0.01).fit(E)
    maps = model.score(E)
    in_sample = np.array(
        [image_score(to_map(maps[i], size=64, sigma=2.0), mode="max") for i in range(E.shape[0])]
    )

    assert held_out.mean() > in_sample.mean()


def test_topk_mode() -> None:
    score_map = torch.zeros(1, 8, 8)
    score_map[0, 0, 0] = 10.0
    assert image_score(to_map(score_map, size=64, sigma=0), mode="max") > 9.0
    assert image_score(to_map(score_map, size=64, sigma=0), mode="topk", topk_frac=0.05) < 5.0


def test_calibration_guards_zero_threshold() -> None:
    calib = make_calibration(np.zeros(10), sigma_k=3.0, review_margin=1.15, k=2)
    assert calib.t_lo > 0
    assert calib.t_hi >= calib.t_lo
