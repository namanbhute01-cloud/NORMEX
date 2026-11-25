# Module 3 — Learning

Everything needed to learn a per-object normal profile from good images only.

| File | Purpose |
|---|---|
| `backbone.py` | Frozen ResNet18, ImageNet weights, layers 1–3, SHA-256 recorded |
| `embedding.py` | Fuse 64+128+256 channels, optional pooling, fixed channel subset |
| `padim.py` | PaDiM fit/score/state; float64 fit, float32 storage |
| `trainer.py` | Good-only orchestration: load → quality → embed → augment → calibrate → fit → save |
| `profile_store.py` | Atomic versioned profiles, `current`, rollback, config/backbone guards |
| `registry.py` | Algorithm plug-in point; PaDiM registered, PatchCore reserved |

Default model: **ResNet18 + PaDiM**. This is the selection document's primary
choice because fitting is closed-form, needs no backpropagation, and produces a
localized anomaly map. Profile size is ~41 MB at `d=100`, `pool=2`.

Challenger: **PatchCore** (own implementation, coreset memory bank). Add it as
`patchcore.py` implementing the same `fit/score/state/from_state` interface and
register it in `registry.py`; do not change M4. Wide-ResNet50-2 and DINOv2 are
later backbone experiments and must pass the latency/RAM gates before adoption.

Dependencies: common, M1, M2, and `module_5_decision.calibrate`. The trainer
refuses paths containing a `test` part or `ground_truth`.

```python
from module_3_learning.trainer import train
from module_3_learning.profile_store import ProfileStore, ProfileMismatchError
```
