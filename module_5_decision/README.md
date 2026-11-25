# Module 5 — Decision

Converts good-only calibration scores into thresholds and an operator verdict.

| File | Purpose |
|---|---|
| `calibrate.py` | k-fold indices, held-out good scores, `Calibration` creation |
| `decide.py` | `GOOD` / `REVIEW` / `DEFECTIVE` / `BAD_IMAGE` decision |

Threshold rule:

```
t_lo = max(mean + sigma_k * std, max(good_scores))
t_hi = t_lo * review_margin
```

Each good image is scored by a fold model that never saw it. No percentile
thresholds on ~10 samples. `BAD_IMAGE` from the M2 quality gate short-circuits
before M4.

Dependencies: common and `module_4_inspection.anomaly_map` utilities.

```python
from module_5_decision.calibrate import kfold_good_scores, make_calibration
from module_5_decision.decide import decide
```
