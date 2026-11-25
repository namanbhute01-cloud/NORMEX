"""T12: decision boundaries and BAD_IMAGE short-circuit."""

from __future__ import annotations

import numpy as np
import torch

from module_0_common.types import Calibration, Inspection, Prepared
from module_5_decision.decide import decide


def _calib() -> Calibration:
    return Calibration(
        t_lo=10.0, t_hi=11.5, good_scores=[9.0, 10.0], mu=9.5, sd=0.5,
        sigma_k=3.0, review_margin=1.15, kfold=2,
    )


def _prep(quality_ok: bool = True) -> Prepared:
    return Prepared(
        tensor=torch.zeros(1, 3, 8, 8),
        display=np.zeros((8, 8, 3), dtype=np.uint8),
        quality_ok=quality_ok,
        quality_msg="" if quality_ok else "too blurry",
    )


def _insp(score: float) -> Inspection:
    return Inspection(score=score, anomaly_map=np.zeros((8, 8), dtype=np.float32), latency_ms=1.0)


def test_status_boundaries() -> None:
    calib = _calib()
    assert decide(_prep(), _insp(9.99), calib, "p", "v001", 5.0).status == "GOOD"
    assert decide(_prep(), _insp(10.0), calib, "p", "v001", 5.0).status == "REVIEW"
    assert decide(_prep(), _insp(11.49), calib, "p", "v001", 5.0).status == "REVIEW"
    assert decide(_prep(), _insp(11.5), calib, "p", "v001", 5.0).status == "DEFECTIVE"


def test_score_norm() -> None:
    verdict = decide(_prep(), _insp(5.0), _calib(), "p", "v001", 5.0)
    assert verdict.score_norm == 0.5


def test_bad_image_short_circuits() -> None:
    verdict = decide(_prep(quality_ok=False), None, _calib(), "p", "v001", 3.0)
    assert verdict.status == "BAD_IMAGE"
    assert verdict.score == 0.0
    assert "blurry" in verdict.message
