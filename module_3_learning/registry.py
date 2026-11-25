"""M3: the only place a new anomaly algorithm gets plugged in."""

from __future__ import annotations

from typing import Any

from module_3_learning.padim import PaDiM

ALGOS: dict[str, type] = {
    "padim": PaDiM,
}


def get_algo(name: str, algo_cfg: dict) -> Any:
    """Instantiate an algorithm by name using its config sub-section."""

    try:
        cls = ALGOS[name]
    except KeyError as exc:
        raise ValueError(f"unknown algo {name!r}; available: {sorted(ALGOS)}") from exc
    options = dict(algo_cfg.get(name, {}) or {})
    return cls(**options)
