"""M5: map a raw inspection result to GOOD / REVIEW / DEFECTIVE / BAD_IMAGE."""

from __future__ import annotations

from typing import Optional

from module_0_common.types import Calibration, Inspection, Prepared, Verdict


def decide(
    prep: Prepared,
    insp: Optional[Inspection],
    calib: Calibration,
    product_id: str,
    version: str,
    total_ms: float,
) -> Verdict:
    """Apply product-specific thresholds.

    ``BAD_IMAGE`` short-circuits before inspection: a blurry or badly lit
    photograph is not evidence about the part.
    """

    if not prep.quality_ok:
        return Verdict(
            product_id=product_id,
            status="BAD_IMAGE",
            score=0.0,
            score_norm=0.0,
            t_lo=calib.t_lo,
            t_hi=calib.t_hi,
            latency_ms=float(total_ms),
            profile_version=version,
            heatmap_path=None,
            message=prep.quality_msg or "image failed the quality gate",
        )

    if insp is None:
        raise ValueError("decide() needs an Inspection for a usable image")

    score = float(insp.score)
    if score < calib.t_lo:
        status = "GOOD"
    elif score < calib.t_hi:
        status = "REVIEW"
    else:
        status = "DEFECTIVE"

    return Verdict(
        product_id=product_id,
        status=status,
        score=score,
        score_norm=score / calib.t_lo if calib.t_lo > 0 else 0.0,
        t_lo=calib.t_lo,
        t_hi=calib.t_hi,
        latency_ms=float(total_ms),
        profile_version=version,
    )
