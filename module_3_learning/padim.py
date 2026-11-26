"""M3: PaDiM — patch distribution modelling with per-position Gaussians.

For every spatial position ``p`` we store the mean feature vector and the
inverse covariance matrix of the normal training patches.  A new patch is
scored with the Mahalanobis distance

    D_p(z) = sqrt((z - mu_p)^T Sigma_p^-1 (z - mu_p))

Fitting is closed-form: no backpropagation, no GPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Positions are processed in blocks to keep peak RAM bounded on a laptop.
_POSITION_CHUNK = 512


class PaDiM:
    name = "padim"

    def __init__(self, eps: float = 0.01) -> None:
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.eps = float(eps)
        self.mean: Optional[torch.Tensor] = None  # [d, P] float32
        self.inv: Optional[torch.Tensor] = None  # [P, d, d] float32
        self.hw: Optional[tuple[int, int]] = None

    def fit(self, E: torch.Tensor) -> "PaDiM":
        """Fit on ``E``: [N, d, H, W] embeddings of good images only."""

        if E.ndim != 4:
            raise ValueError(f"expected [N,d,H,W] embeddings, got {tuple(E.shape)}")
        N, d, H, W = E.shape
        if N < 2:
            raise ValueError("PaDiM needs at least 2 training embeddings")

        P = H * W
        X = E.detach().reshape(N, d, P).to(torch.float64)
        mean = X.mean(dim=0)  # [d,P]
        Xc = X - mean

        cov = torch.empty((P, d, d), dtype=torch.float64)
        for start in range(0, P, _POSITION_CHUNK):
            stop = min(start + _POSITION_CHUNK, P)
            block = Xc[:, :, start:stop].permute(2, 0, 1)  # [c,N,d]
            cov[start:stop] = torch.bmm(block.transpose(1, 2), block) / (N - 1)

        cov += self.eps * torch.eye(d, dtype=torch.float64)

        inv = torch.empty_like(cov)
        for start in range(0, P, _POSITION_CHUNK):
            stop = min(start + _POSITION_CHUNK, P)
            inv[start:stop] = torch.linalg.inv(cov[start:stop])

        if not torch.isfinite(inv).all():
            raise ValueError(
                "PaDiM covariance inversion produced NaN/Inf. "
                "Increase algo.padim.eps or add more (varied) good images."
            )

        self.mean = mean.float().contiguous()
        self.inv = inv.float().contiguous()
        self.hw = (H, W)
        return self

    @torch.inference_mode()
    def score(self, E: torch.Tensor) -> torch.Tensor:
        """Score ``E``: [B, d, H, W] -> [B, H, W] Mahalanobis distances."""

        if self.mean is None or self.inv is None or self.hw is None:
            raise RuntimeError("PaDiM is not fitted; call fit() or from_state() first")
        if E.ndim != 4:
            raise ValueError(f"expected [B,d,H,W] embeddings, got {tuple(E.shape)}")
        B, d, H, W = E.shape
        if (H, W) != self.hw:
            raise ValueError(f"embedding grid {(H, W)} does not match fitted grid {self.hw}")
        if d != self.mean.shape[0]:
            raise ValueError(f"embedding dim {d} does not match fitted dim {self.mean.shape[0]}")

        X = E.reshape(B, d, -1).float() - self.mean  # [B,d,P]
        transformed = torch.einsum("bdp,pde->bpe", X, self.inv)
        squared = (transformed * X.permute(0, 2, 1)).sum(dim=-1)  # [B,P]
        squared = squared.clamp_min(0.0)
        # MKL's VML sqrt dispatches an AVX-encoded kernel on some non-AVX
        # Celerons (SIGILL in mkl_vml_kernel_sSqrt).  NumPy's sqrt is safe there
        # and the transfer is negligible for these small [B,P] maps.
        if squared.device.type == "cpu":
            return torch.from_numpy(np.sqrt(squared.numpy())).reshape(B, H, W)
        return squared.sqrt().reshape(B, H, W)

    def state(self) -> dict[str, np.ndarray]:
        """Serializable float32 state (``mean`` [d,P], ``inv_cov`` [P,d,d])."""

        if self.mean is None or self.inv is None:
            raise RuntimeError("PaDiM is not fitted")
        return {
            "mean": self.mean.cpu().numpy().astype(np.float32, copy=True),
            "inv_cov": self.inv.cpu().numpy().astype(np.float32, copy=True),
        }

    @classmethod
    def from_state(cls, state: dict, eps: float, hw: tuple[int, int]) -> "PaDiM":
        model = cls(eps=eps)
        mean = np.asarray(state["mean"], dtype=np.float32)
        inv = np.asarray(state["inv_cov"], dtype=np.float32)
        if mean.ndim != 2 or inv.ndim != 3:
            raise ValueError("bad PaDiM state shapes")
        if inv.shape[0] != mean.shape[1] or inv.shape[1] != inv.shape[2] or inv.shape[1] != mean.shape[0]:
            raise ValueError("PaDiM state is inconsistent (mean vs inv_cov)")
        if inv.shape[0] != int(hw[0]) * int(hw[1]):
            raise ValueError("PaDiM state position count does not match hw")
        model.mean = torch.from_numpy(mean.copy()).contiguous()
        model.inv = torch.from_numpy(inv.copy()).contiguous()
        model.hw = (int(hw[0]), int(hw[1]))
        return model

    @property
    def nbytes(self) -> int:
        if self.mean is None or self.inv is None:
            return 0
        return int(self.mean.numel() * 4 + self.inv.numel() * 4)
