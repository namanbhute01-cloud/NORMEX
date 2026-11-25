"""T4: embedding fusion and channel selection."""

from __future__ import annotations

import pytest
import torch

from module_3_learning.embedding import embed, make_channel_idx


def _features(batch: int = 2, size: int = 256):
    return (
        torch.randn(batch, 64, size // 4, size // 4),
        torch.randn(batch, 128, size // 8, size // 8),
        torch.randn(batch, 256, size // 16, size // 16),
    )


def test_embedding_shape() -> None:
    channel_idx = make_channel_idx(448, 100, seed=0)
    out = embed(_features(), channel_idx, pool=2)
    assert tuple(out.shape) == (2, 100, 32, 32)


def test_channel_idx_is_deterministic_and_valid() -> None:
    first = make_channel_idx(448, 100, seed=7)
    second = make_channel_idx(448, 100, seed=7)
    other = make_channel_idx(448, 100, seed=8)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.numel() == 100
    assert torch.all(first[1:] > first[:-1])  # sorted + unique
    assert int(first.min()) >= 0 and int(first.max()) < 448


def test_bad_pool_raises() -> None:
    channel_idx = make_channel_idx(448, 10, seed=0)
    with pytest.raises(ValueError, match="pool"):
        embed(_features(), channel_idx, pool=3)  # 64 is not divisible by 3


def test_bad_dimension_raises() -> None:
    with pytest.raises(ValueError):
        make_channel_idx(448, 0, seed=0)
    with pytest.raises(ValueError):
        make_channel_idx(448, 449, seed=0)
