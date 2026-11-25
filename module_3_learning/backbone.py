"""M3: frozen ImageNet ResNet18 feature extractor (layers 1-3).

The classifier head and ``layer4`` are intentionally dropped: PaDiM only
needs the mid-level feature maps.  The network is never trained, never put
in train mode, and never allowed to fetch weights from the network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Backbone(nn.Module):
    """Frozen ResNet18 stem + layer1-3.

    Shapes for ``S=256``:
        f1: [B,  64, 64, 64]
        f2: [B, 128, 32, 32]
        f3: [B, 256, 16, 16]
    """

    out_channels: tuple[int, int, int] = (64, 128, 256)

    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"ResNet18 weights not found at {path}. "
                "Run `python scripts/fetch_weights.py` once while online, then retry offline."
            )

        self.sha256 = sha256_file(path)
        model = resnet18(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"weights at {path} do not match torchvision resnet18: {exc}"
            ) from exc

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.l1 = model.layer1
        self.l2 = model.layer2
        self.l3 = model.layer3

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "Backbone":
        """Keep the backbone in eval mode no matter what the caller asks.

        BatchNorm must use its frozen running statistics; flipping to train
        mode would silently change every feature.
        """

        return super().train(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x``: [B,3,S,S] -> (f1, f2, f3) feature maps."""

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,S,S] input, got {tuple(x.shape)}")
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        return f1, f2, f3
