"""Small timing helpers used by the trainer, inspector and CLI."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Context manager that records elapsed milliseconds in ``self.ms``."""

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0


class Stages:
    """Accumulate per-stage timings into a plain dict of milliseconds."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = (time.perf_counter() - t0) * 1000.0

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms
