# Camera Inspector — module map

The project is split into one folder per module. Folder names are Python-safe
versions of the architecture names: `module_1_acquisition` is "Module 1 —
Acquisition", and so on.

```
main.py                     composition root: CLI, joins all modules
configs/                    YAML configuration
module_0_common/            shared contracts, config, seed, timing
module_1_acquisition/       M1: file/folder loader + webcam + mobile IP camera
module_2_preprocessing/     M2: quality, ROI, alignment, letterbox/normalize
module_3_learning/          M3: backbone, embedding, PaDiM, trainer, profiles
module_4_inspection/        M4: anomaly map, heatmap, inspector
module_5_decision/          M5: k-fold calibration, verdicts
module_6_evaluation/        M6: MVTec adapter, metrics, synthetic demo parts
```

`scripts/` holds one-off tooling. `tests/` holds unit, guard and synthetic
end-to-end tests. `profiles/`, `pretrained/`, `datasets/` and `results/` hold
generated or downloaded artifacts, not source code.

## Module responsibilities

| Folder | Architecture name | Responsibility | Key entry points |
|---|---|---|---|
| `module_0_common` | Shared | Typed contracts, YAML config + hash, seeding, timing | `types.py`, `config.py`, `timing.py`, `seed.py` |
| `module_1_acquisition` | M1 Acquisition | Turn files, folders, webcam and IP-camera frames into `Frame` objects | `loader.py::list_images`, `load_frame`; `camera.py::Camera`, `SnapshotCamera`, `capture_session`, `resolve_url` |
| `module_2_preprocessing` | M2 Preprocess | Quality gate, fixed ROI, optional alignment/denoise/illumination, letterbox/normalize, heatmap geometry | `pipeline.py::preprocess`, `quality.py`, `roi.py`, `align.py`, `transform.py::map_to_source`, `augment.py` |
| `module_3_learning` | M3 Learning | Frozen backbone, embedding, PaDiM, training, versioned profiles | `backbone.py`, `embedding.py`, `padim.py`, `trainer.py`, `profile_store.py`, `registry.py` |
| `module_4_inspection` | M4 Inspection | Anomaly map, smoothing, heatmap, one-photo inspection | `anomaly_map.py`, `inspector.py` |
| `module_5_decision` | M5 Decision | k-fold good scores, thresholds, GOOD/REVIEW/DEFECTIVE | `calibrate.py`, `decide.py` |
| `module_6_evaluation` | Eval / benchmark | MVTec adapter, AUROC/FAR/FRR/P95, deterministic synthetic demo parts, validation support | `mvtec.py`, `metrics.py`, `synthetic.py` |

## How the modules join

```
main.py
  ├─ module_1_acquisition.loader/camera → Frame
  ├─ module_2_preprocessing.pipeline  → Prepared (tensor + geometry)
  ├─ module_3_learning.trainer        → profile (train mode)
  │    └─ module_3_learning.profile_store
  ├─ module_4_inspection.inspector    → Inspection (score + heatmap)
  ├─ module_5_decision.decide         → Verdict
  └─ module_6_evaluation              → offline metrics only
```

Dependency direction:

- `module_0_common` imports no other project module.
- M1 and M2 depend on common only.
- M3 depends on common, M1, M2 and `module_5_decision.calibrate`.
- M4 depends on common and M3 (embedding).
- M5 depends on common and M4 (`anomaly_map` utilities).
- M6 may read M1/M3/M4/M5, but **learning/inspection production modules never
  import M6**; the guard test enforces this. `main.py` only wires M6 for the
  explicit `validate` command.
- `main.py` is the only composition root.

## Model and package choices

These follow `CAMERA_INSPECTOR_MODEL_SELECTION_AND_VALIDATION.md`.

| Module | Default choice | Challenger / next | Status |
|---|---|---|---|
| M1 | OpenCV file loading, local webcam, IP-camera MJPEG/RTSP stream, JPEG snapshot polling | vendor SDK behind `camera.py` | Loader + webcam + IP camera done; vendor SDK later |
| M2 | OpenCV classical CV: Laplacian quality, fixed ROI, letterbox, normalization | ECC alignment now available but off by default; ORB/RANSAC later if needed | Quality + ROI + optional alignment/denoise/illumination + letterbox done; automatic segmentation deferred |
| M3a | **ResNet18** ImageNet, layers 1–3, frozen | Wide-ResNet50-2, then DINOv2 ViT-S/14 | ResNet18 done |
| M3b | **PaDiM** (own implementation, closed-form) | **PatchCore** (coreset memory bank) | PaDiM done; PatchCore not built |
| M4 | PyTorch CPU `inference_mode` + OpenCV heatmap | ONNX Runtime / OpenVINO if P95 ≥ 1 s | Done on CPU |
| M5 | k-fold good scores; `max(μ + 3σ, max)` thresholds | split-conformal alternative | Done |
| M6 | MVTec adapter + metrics | paired bootstrap, perturbation tests | Adapter/metrics done; full benchmark matrix pending |

**Do not swap the default yet.** The selection document says: prove ResNet18 +
PaDiM first; adopt Wide-ResNet50-2, PatchCore or DINOv2 only if a measured,
paired-bootstrap improvement survives the latency/RAM gates. EfficientAD is
rejected for this phase because it needs backprop training and its published
latency is GPU-based.

## Naming rule

Python package names cannot contain spaces or start with a digit, so the
folders use `module_<n>_<name>`:

| Human name | Folder |
|---|---|
| Module 0 — Common | `module_0_common` |
| Module 1 — Acquisition | `module_1_acquisition` |
| Module 2 — Preprocessing | `module_2_preprocessing` |
| Module 3 — Learning | `module_3_learning` |
| Module 4 — Inspection | `module_4_inspection` |
| Module 5 — Decision | `module_5_decision` |
| Module 6 — Evaluation | `module_6_evaluation` |
