# NOTES — decisions, deviations, open questions

Phase 1 scope: M1 (file loader + webcam + mobile IP camera) + M2 (quality/ROI/letterbox/optional
alignment and photometry/augment) + M3
(backbone/embedding/PaDiM/trainer/profile store) + M4 (inspection/heatmap) +
M5 (calibration/decision) + CLI + synthetic and MVTec validation harness.

## Decisions

- **PaDiM is the Phase-1 algorithm.** It needs no backpropagation, fits from
  closed-form statistics, and gives a per-position anomaly map. The registry
  (`module_3_learning/registry.py`) is the single plug-in point for PatchCore
  later.
- **No percentile thresholds.** Calibration uses k-fold held-out good scores
  and `t_lo = max(mu + sigma_k*sd, max(good_scores))`, `t_hi = t_lo * margin`.
- **The final model is fitted on all originals + augmented copies**, while
  each calibration fold sees fewer images. The final model therefore scores
  good images slightly lower than the held-out calibration scores, making
  thresholds mildly conservative. This is intentional; it is not "fixed".
- **Quality limits are learned from the good set** and stored per profile.
  During training the gate is not applied (the training images define normal).
  Blur, brightness, saturation, foreground occupancy and image size are
  measured on the ROI, not on the background.
- **The image is letterboxed, never stretched.** A non-square photo is resized
  to fit the model input and padded; the scale/padding is stored in the
  in-memory geometry so the heatmap can be mapped back to the original photo.
- **A profile owns its preprocessing settings.** ROI, alignment, denoise,
  illumination, letterbox and quality limits are written to `preprocess.json`
  and used at inspection time; the current YAML cannot silently change them.
- **ROI, alignment, denoise and illumination are off by default.** The fixed
  jig is the primary pose control. Software alignment is conservative ECC
  (translation + rotation only) and refuses large corrections; denoise and
  illumination correction are measured fallbacks because they can erase a
  defect. Automatic segmentation is deferred.
- **The CLI heatmap is drawn on the original photograph**, not on the 256×256
  model view, using `transform.map_to_source()`.
- **`input.size` must be a multiple of 32** so ResNet strides and the optional
  embedding pool divide cleanly.
- **Profile loads require the expected config hash and backbone SHA-256.**
  There is no silent-load path.
- **Saves are atomic**: a temporary version directory is renamed into place,
  then `current` is updated. A failure leaves the previous version active.
- **Patch scores are upsampled to the profile input size and Gaussian-smoothed
  before reduction**; calibration and inspection use exactly the same path.
- **Heatmaps use a fixed `vmax = t_hi * 1.5`**, never per-image min/max, so
  colours are comparable across photographs.
- **Acquisition is camera-agnostic.** Local webcam (`Camera`, OpenCV
  `VideoCapture`), IP camera MJPEG stream (`Camera(url=...)`), or IP camera
  JPEG snapshot (`SnapshotCamera(url=...)`). `--ip` builds
  `http://<ip>:<port>/video`; `--snapshot` changes it to `/shot.jpg`.
  Snapshot polling is preferred because it is a fresh request per frame.
  Credentials in URLs are converted to HTTP Basic in snapshot mode and
  redacted before logging. Network access is only to the camera URL the user
  passes; no cloud service is contacted.

## Deviations / additions (smallest safe option per working rule 10)

- Added `tests/test_preprocess.py`. T2/T8 require quality/transform/augment
  tests but the prescribed file list had no natural home for them.
- `module_5_decision/calibrate.py::image_scores` takes an explicit `size`
  argument (the spec's signature omitted it). Calibration must upsample to the
  same size as inspection, otherwise thresholds are not comparable.
- `train()` refuses path parts equal to `test` or containing `ground_truth`
  (case-insensitive), rather than every substring `test` (which would reject
  legitimate names such as `testing_parts`).
- `requirements.txt` pins the versions actually installed in this environment.

## Environment used for the measurements in this repo

- CPU: Intel Celeron N4500 @ 1.10 GHz (2 cores, no GPU)
- RAM: 7 GB
- Python 3.12.3, torch 2.15.0+cpu, torchvision 0.29.2+cpu
- **Caveat:** this machine was under heavy background load (load average
  ~7 on 2 cores) during the runs. Latency numbers here are pessimistic and
  are not a claim about a normal idle laptop. Always re-measure on the target
  machine with `scripts/run_validation.py`.

## Measured results

M2 preprocessing-only latency measured on this 2-core Celeron N4500 (loaded
dev box, not the target laptop):

| Input | Mean | P95 | Notes |
|---|---:|---:|---|
| 256×256 synthetic | 4.5 ms | 9.1 ms | n=50, quality gate enabled |
| 1920×1080 synthetic | 62.3 ms | 72.6 ms | n=15; ~45 ms is the Laplacian quality check |

These are preprocessing-only numbers. They are not the end-to-end P95, which
also includes the backbone, scoring and heatmap.

See `results/` for the real artifacts. The synthetic end-to-end test passes
all of its gates: train **43.0 s** (spec dev gate < 120 s), AUROC **1.0000**,
defect mean 38.67 vs good mean 4.03, 15/15 defects flagged, 0/10 unseen good
marked DEFECTIVE, localization within 24 px, heatmap written. The full suite
took **63 s** in a clean run, **50–116 s** in recent runs after the M2 and
camera additions, and **148–165 s** under heavy contention (63 passed every
time; the
per-test 120 s training gate still passed). Under extreme background load the
same synthetic training was
measured at 145–150 s; that is environment contention, not a code-path
change, and those runs still met the 600 s product budget. The spec's
< 2 min whole-suite guideline is met only on the quieter run.

MVTec AD was **not available locally**, so gates G3-G7 have not been run and
no MVTec AUROC/FAR/FRR numbers are quoted anywhere in this repository. Run:

```
python main.py validate --category bottle
python main.py validate --category all
```

once the dataset is placed under `datasets/mvtec_ad/`.

## Known limitations (by design of Phase 1)

- **Pose sensitivity.** PaDiM models each spatial position independently.
  A fixed camera/jig is assumed. Varying pose will raise good-image scores.
- **Small defects.** At 256x256 a defect smaller than one feature cell
  (~8 px at the 32x32 grid) can disappear. A 192x192 run kept AUROC at 1.0
  but failed the 24 px localization check on one scratch (peak 36.4 px);
  resolution reduction is not accuracy-neutral. 320/tiling is a later
  experiment.
- **Logical defects** (missing/misplaced components) are harder than surface
  defects; this phase targets structural/appearance anomalies.
- **Lighting drift** between train and inspect is the most common cause of
  false rejects. The quality gate and mild brightness augmentation only
  partly compensate.
- **Conservative thresholds** because the final fit sees all data (above).
- **No UI, ONNX/OpenVINO, SQLite, multi-view or vendor camera SDK** in this
  phase. Laptop/USB webcam capture through OpenCV is implemented.

## Open questions

- MVTec `bottle` AUROC and P95 latency on the real target laptop (G3/G5).
- Whether 320x320 or ROI crops are needed for the smallest defects of the
  actual 15 objects.
- Whether PatchCore is needed for pose-varying objects (`screw`,
  `metal_nut`-like categories).
