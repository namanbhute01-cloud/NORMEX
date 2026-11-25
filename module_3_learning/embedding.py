"""M3: fuse layer 1-3 feature maps into one patch-embedding map."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_channel_idx(total: int, d: int, seed: int) -> torch.Tensor:
    """Pick ``d`` deterministic, sorted, unique channel indices out of ``total``.

    PaDiM's random feature selection keeps the per-position covariance small
    enough to store and invert on a CPU.
    """

    if total < 1:
        raise ValueError("total must be >= 1")
    if not 1 <= d <= total:
        raise ValueError(f"d must be in [1, {total}], got {d}")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:d].sort().values.to(torch.int64)


def embed(feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
          channel_idx: torch.Tensor, pool: int = 1) -> torch.Tensor:
    """Fuse ``(f1, f2, f3)`` -> [B, d, H, W].

    ``f2``/``f3`` are bilinearly upsampled to ``f1``'s grid, concatenated
    (64+128+256 = 448 channels), optionally average-pooled, then reduced to
    the channels chosen at training time.
    """

    if pool < 1:
        raise ValueError(f"pool must be >= 1, got {pool}")
    f1, f2, f3 = feats
    size = f1.shape[-2:]

    def _up(tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    fused = torch.cat([f1, _up(f2), _up(f3)], dim=1)  # [B,448,S/4,S/4]
    if pool > 1:
        if fused.shape[-2] % pool != 0 or fused.shape[-1] % pool != 0:
            raise ValueError(
                f"pool={pool} does not divide feature map {tuple(fused.shape[-2:])}"
            )
        fused = F.avg_pool2d(fused, pool)
    return fused.index_select(1, channel_idx.to(fused.device))
