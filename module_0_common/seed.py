"""Deterministic seeding for every library that uses randomness."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed ``random``, NumPy and PyTorch (CPU)."""

    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
