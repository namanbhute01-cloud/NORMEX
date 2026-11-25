# Module 0 — Common

Shared contracts and utilities. This module must not import any other project
module.

| File | Purpose |
|---|---|
| `types.py` | `Frame`, `Prepared`, `Inspection`, `Verdict`, `Calibration` dataclasses |
| `config.py` | YAML load/validation, dotted overrides, `config_hash`, path resolution |
| `timing.py` | `Stages` per-stage `perf_counter` timings |
| `seed.py` | Deterministic Python/NumPy/PyTorch seeding |

Import examples:

```python
from module_0_common.types import Frame, Prepared
from module_0_common.config import load_config, config_hash
```
