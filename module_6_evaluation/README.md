# Module 6 — Evaluation

Offline measurement only. This is the only place allowed to read MVTec
`test/` and `ground_truth/` data.

| File | Purpose |
|---|---|
| `mvtec.py` | Category discovery, test-image listing, mask loading |
| `metrics.py` | Image AUROC, pixel AUROC, FAR/FRR, P95 |
| `synthetic.py` | Deterministic synthetic demo parts (not a benchmark) |

Benchmark entry point: `scripts/run_validation.py` (also exposed as
`python main.py validate`). It writes timestamped JSON + CSV under `results/`.
The full model-selection matrix (PaDiM vs PatchCore, R18 vs WRN50 vs DINOv2,
224/256/320, coreset ratios, paired bootstrap) is not implemented yet.

Production modules must not import this module; `tests/test_guards.py`
enforces that rule.

```python
from module_6_evaluation import mvtec, metrics
```
