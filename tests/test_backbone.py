"""T3: frozen ResNet18 backbone behaviour."""

from __future__ import annotations

import pytest
import torch

from module_3_learning.backbone import Backbone


def test_output_shapes(backbone: Backbone) -> None:
    x = torch.zeros(2, 3, 256, 256)
    f1, f2, f3 = backbone(x)
    assert tuple(f1.shape) == (2, 64, 64, 64)
    assert tuple(f2.shape) == (2, 128, 32, 32)
    assert tuple(f3.shape) == (2, 256, 16, 16)


def test_parameters_frozen(backbone: Backbone) -> None:
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_train_mode_cannot_flip_batchnorm(backbone: Backbone) -> None:
    backbone.train()
    assert backbone.training is False
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.training is False


def test_forward_is_deterministic(backbone: Backbone) -> None:
    x = torch.randn(1, 3, 256, 256)
    first = backbone(x)
    second = backbone(x)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_missing_weights_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_weights"):
        Backbone(tmp_path / "does_not_exist.pth")
