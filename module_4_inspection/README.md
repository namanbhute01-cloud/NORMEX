# Module 4 — Inspection

Turns one `Prepared` photograph into an anomaly score and a heatmap.

| File | Purpose |
|---|---|
| `anomaly_map.py` | Upsample patch scores, Gaussian smooth, reduce to image score, colour overlay |
| `inspector.py` | Loaded backbone + profile → `Inspection` (score, map, per-stage latency) |

Image-score modes: `max` (default, sensitive to small defects) and `topk`
(mean of the top fraction, more noise-robust). Heatmaps use a fixed
`vmax = t_hi * 1.5`, never per-image min/max, so colours are comparable.

Dependencies: common and `module_3_learning.embedding`; the algorithm-specific
scoring stays behind the profile's `algo.score()`.

```python
from module_4_inspection.inspector import Inspector
from module_4_inspection.anomaly_map import to_map, image_score, overlay
```
