"""T5: PaDiM mathematics and state handling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from module_3_learning.padim import PaDiM


def _gaussian_features(n: int = 40, d: int = 16, h: int = 8, w: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    location = torch.randn(d, 1, 1, generator=generator)
    scale = torch.rand(d, 1, 1, generator=generator) + 0.5
    return location + scale * torch.randn(n, d, h, w, generator=generator)


def test_in_distribution_score_scale() -> None:
    E = _gaussian_features()
    model = PaDiM(eps=0.01).fit(E)
    scores = model.score(E)
    assert tuple(scores.shape) == (40, 8, 8)
    mean = float(scores.mean())
    assert 0.5 * np.sqrt(16) < mean < 1.5 * np.sqrt(16)


def test_far_features_score_much_higher() -> None:
    E = _gaussian_features()
    model = PaDiM(eps=0.01).fit(E)
    in_distribution = float(model.score(E).mean())
    far = model.score(E[0:1] + 10.0).mean()
    assert float(far) > 3.0 * in_distribution


def test_score_of_training_mean_is_near_zero() -> None:
    E = _gaussian_features()
    model = PaDiM(eps=0.01).fit(E)
    mean_embedding = model.mean.reshape(1, 16, 8, 8)
    assert float(model.score(mean_embedding).max()) < 1e-3


def test_matches_scipy_reference() -> None:
    E = _gaussian_features(n=20, d=8, seed=3)
    eps = 0.05
    model = PaDiM(eps=eps).fit(E)

    X = E.reshape(20, 8, -1).double().numpy()
    mean = X.mean(axis=0)
    centered = X - mean
    cov = np.einsum("ndp,nep->pde", centered, centered) / (20 - 1)
    cov += eps * np.eye(8)
    inv = np.linalg.inv(cov)

    diff = X[0] - mean  # [d,P]
    m2 = np.einsum("dp,pde,ep->p", diff, inv, diff)
    reference = np.sqrt(np.clip(m2, 0, None)).reshape(8, 8)

    actual = model.score(E[0:1]).numpy().reshape(8, 8)
    assert np.max(np.abs(actual - reference)) < 1e-3


def test_single_image_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        PaDiM().fit(torch.randn(1, 8, 4, 4))


def test_state_round_trip() -> None:
    E = _gaussian_features(n=20, d=8, seed=5)
    model = PaDiM(eps=0.02).fit(E)
    clone = PaDiM.from_state(model.state(), eps=0.02, hw=model.hw)
    assert torch.allclose(model.score(E), clone.score(E), atol=1e-6)


def test_grid_mismatch_raises() -> None:
    model = PaDiM().fit(_gaussian_features(n=10, d=8, h=4, w=4))
    with pytest.raises(ValueError, match="grid"):
        model.score(torch.randn(1, 8, 5, 5))
