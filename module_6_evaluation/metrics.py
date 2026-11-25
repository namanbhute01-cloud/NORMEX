"""Evaluation metrics.  Numbers are only ever computed from real runs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    """Threshold-free ranking quality; ``nan`` if only one class is present."""

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(
    maps: list[np.ndarray],
    masks: list[np.ndarray],
    max_pixels: int = 200_000,
    seed: int = 0,
) -> float:
    """Pixel-level AUROC over anomaly maps vs ground-truth masks.

    Large categories are subsampled deterministically to keep this fast.
    """

    if not maps:
        return float("nan")
    flat_scores = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    flat_labels = np.concatenate([(np.asarray(m) > 0).astype(np.uint8).ravel() for m in masks])
    if np.unique(flat_labels).size < 2:
        return float("nan")

    if flat_scores.size > max_pixels:
        rng = np.random.default_rng(seed)
        index = rng.choice(flat_scores.size, size=max_pixels, replace=False)
        flat_scores = flat_scores[index]
        flat_labels = flat_labels[index]
        if np.unique(flat_labels).size < 2:
            return float("nan")

    return float(roc_auc_score(flat_labels, flat_scores))


def far_frr(statuses: list[str], labels: list[int]) -> dict:
    """False-accept / false-reject / review rates.

    FAR: defective image marked GOOD (the costly manufacturing error).
    FRR: good image marked DEFECTIVE.
    """

    statuses = np.asarray(statuses, dtype=object)
    labels = np.asarray(labels, dtype=int)
    good = labels == 0
    defect = labels == 1

    def _rate(mask: np.ndarray, status: str) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(statuses[mask] == status))

    return {
        "far": _rate(defect, "GOOD"),
        "frr": _rate(good, "DEFECTIVE"),
        "review_rate_good": _rate(good, "REVIEW"),
        "review_rate_defect": _rate(defect, "REVIEW"),
    }


def p95(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95))
