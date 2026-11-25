"""M5: threshold calibration from held-out *good* images only.

A percentile on ~10 calibration images is meaningless.  Instead every good
image is scored by a model that never saw it (k-fold), producing one honest
"normal" score per image.  Thresholds are then ``mean + k*sd`` with a floor
at the worst held-out score.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from module_0_common.types import Calibration
from module_4_inspection.anomaly_map import image_score, to_map

MIN_GOOD_IMAGES = 10


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition ``range(n)`` into ``k`` folds with no train/val overlap."""

    if n < 2:
        raise ValueError("k-fold needs at least 2 samples")
    if k < 2:
        raise ValueError("kfold must be >= 2")
    k = min(k, n)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    folds = np.array_split(permutation, k)
    all_indices = np.arange(n)
    return [
        (np.setdiff1d(all_indices, np.sort(fold)), np.sort(fold))
        for fold in folds
    ]


def image_scores(
    score_maps: torch.Tensor,
    size: int,
    sigma: float,
    mode: str,
    topk_frac: float,
) -> np.ndarray:
    """Reduce a batch of patch-score maps ``[B,h,w]`` to one score per image."""

    maps = score_maps.detach().cpu().numpy() if torch.is_tensor(score_maps) else np.asarray(score_maps)
    if maps.ndim != 3:
        raise ValueError(f"expected [B,h,w] score maps, got {maps.shape}")
    return np.array(
        [image_score(to_map(maps[i], size=size, sigma=sigma), mode=mode, topk_frac=topk_frac)
         for i in range(maps.shape[0])],
        dtype=np.float64,
    )


def kfold_good_scores(
    E_orig: torch.Tensor,
    aug_index: Optional[Sequence[Sequence[int]]],
    E_aug: Optional[torch.Tensor],
    algo_factory: Callable[[], object],
    k: int,
    seed: int,
    *,
    size: int,
    sigma: float,
    mode: str,
    topk_frac: float,
) -> np.ndarray:
    """One held-out score per good image, aligned with ``E_orig`` order.

    Each fold is fitted on the training originals **plus their augmented
    copies**; held-out images are scored un-augmented.
    """

    n = int(E_orig.shape[0])
    if n < MIN_GOOD_IMAGES:
        raise ValueError(f"need at least {MIN_GOOD_IMAGES} good images, got {n}")

    scores = np.full(n, np.nan, dtype=np.float64)
    for train_idx, val_idx in kfold_indices(n, k, seed):
        parts = [E_orig[train_idx]]
        if E_aug is not None and aug_index is not None:
            augmented = [E_aug[list(aug_index[i])] for i in train_idx if len(aug_index[i]) > 0]
            if augmented:
                parts.append(torch.cat(augmented, dim=0))
        train_embeddings = torch.cat(parts, dim=0)

        algo = algo_factory()
        algo.fit(train_embeddings)
        fold_scores = image_scores(
            algo.score(E_orig[val_idx]), size=size, sigma=sigma, mode=mode, topk_frac=topk_frac
        )
        scores[val_idx] = fold_scores

    if np.isnan(scores).any():  # pragma: no cover - defensive
        raise RuntimeError("k-fold calibration left some images unscored")
    return scores


def make_calibration(
    scores: np.ndarray,
    sigma_k: float,
    review_margin: float,
    k: int,
) -> Calibration:
    """Build thresholds from held-out good scores."""

    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot calibrate on zero scores")
    if review_margin < 1.0:
        raise ValueError("review_margin must be >= 1.0")

    mu = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    t_lo = max(mu + float(sigma_k) * sd, float(values.max()))
    t_lo = max(t_lo, 1e-6)  # keep score_norm finite
    t_hi = max(t_lo * float(review_margin), t_lo)

    return Calibration(
        t_lo=float(t_lo),
        t_hi=float(t_hi),
        good_scores=[float(v) for v in values],
        mu=mu,
        sd=sd,
        sigma_k=float(sigma_k),
        review_margin=float(review_margin),
        kfold=int(k),
    )
