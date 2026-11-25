# Camera Inspector — Phase 1

Good-only visual inspection for manufactured parts. Learn what a correct
part looks like from **good photographs only**, then flag deviations on new
photographs — offline, CPU-only, no GPU.

**Pipeline:** frozen ImageNet ResNet18 (layers 1-3) → per-position patch
statistics (**PaDiM**) → anomaly heatmap → calibrated
`GOOD / REVIEW / DEFECTIVE / BAD_IMAGE` verdict.

The code is split into one top-level folder per module
(`module_0_common` … `module_6_evaluation`). See `MODULES.md` for the module
map, dependency rules, and the model-selection rationale.

This phase contains the learning engine, the inspection engine, file/folder
acquisition, laptop/USB webcam capture and mobile IP camera capture. There is
no UI, database, ONNX/OpenVINO or PatchCore yet; those are later phases.
PatchCore has a single plug-in point (`module_3_learning/registry.py`).

**First run?** Follow `DEMO_GUIDE.md` for the full fresh-start sequence:
synthetic data, mobile IP camera and the laptop-webcam fallback.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch first (one time)
pip install torch==2.15.0+cpu torchvision==0.29.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# One-time weights fetch (the only network step)
python scripts/fetch_weights.py

# Reproducible demo: synthetic parts -> train -> inspect -> summary
python scripts/make_demo_dataset.py --out datasets/demo
python main.py train --product demo --images datasets/demo/good
python main.py inspect --product demo --image datasets/demo/new_good \
    --out results/demo/good --jsonl results/demo/good.jsonl
python main.py inspect --product demo --image datasets/demo/defect \
    --out results/demo/defect --jsonl results/demo/defect.jsonl
python scripts/summarize_inspect.py \
    --good results/demo/good.jsonl --defect results/demo/defect.jsonl

# Use a phone as the camera over local Wi-Fi (IP Webcam app)
python main.py capture --ip 192.168.1.23 --port 8080 --snapshot \
    --out datasets/ipcam/good --count 30 --interval 0.5
python main.py train --product ipcam_part --images datasets/ipcam/good
python main.py capture --ip 192.168.1.23 --port 8080 --snapshot \
    --out datasets/ipcam/test --count 5 --prefix test
python main.py inspect --product ipcam_part --image datasets/ipcam/test

# Fallback: laptop webcam
python main.py capture --list
python main.py capture --out datasets/camera/good --count 30 --interval 0.5
python main.py train --product camera_part --images datasets/camera/good

# Profile management
python main.py list
python main.py rollback --product demo
```

No cloud service is used: weights are loaded from `pretrained/resnet18.pth`,
and camera frames come only from the URL you pass. The IP camera is expected
to be on your local network.

## What "learning" means here

No defect images are used. For each product:

1. Good images are embedded with the frozen backbone.
2. Defect-safe augmentation (small shift/rotation/brightness) adds variation.
3. PaDiM stores a mean vector and inverse covariance at every spatial
   position.
4. Each good image is scored by a k-fold model that never saw it; thresholds
   are `t_lo = max(mu + k*sd, max(good score))`, `t_hi = t_lo * margin`.
5. The final model is fitted on all originals + augmented copies and saved as
   a versioned profile.

Inspection measures the Mahalanobis distance of each patch to the learned
normal distribution, smooths the map, and reduces it to an image score.

## CLI

```
python main.py train    --product ID --images DIR [--config configs/default.yaml]
python main.py capture  --list
python main.py capture  --ip PHONE_IP [--port 8080] [--snapshot] --out DIR
python main.py capture  --url http://PHONE_IP:8080/video --out DIR
python main.py capture  --snapshot-url http://PHONE_IP:8080/shot.jpg --out DIR
python main.py capture  --out DIR [--camera 0] [--count 30] [--interval 0.5] [--preview]
python main.py inspect  --product ID --image PATH_OR_DIR [--out results/inspect]
                        [--jsonl FILE] [--no-heatmap]
python main.py list
python main.py rollback --product ID
python main.py validate --category bottle|all [--dataset datasets/mvtec_ad]
```

`capture` supports a local webcam, an IP-camera MJPEG/RTSP stream, or JPEG
snapshot polling. `--snapshot` is recommended for phones because every frame
is a fresh request. `inspect` prints one JSON line per image, optionally writes
the same JSON to `--jsonl`, and saves `<name>_heatmap.png` at the original
photo resolution. Exit code is 2 for usage/profile/camera errors.

## Configuration

`configs/default.yaml` holds the whole Phase-1 configuration: input size,
embedding dimension/pooling, PaDiM epsilon, augmentation, calibration and
quality gate. Unknown keys are rejected; `config_hash` covers the keys that
affect feature compatibility, and a profile refuses to load under a different
hash or backbone SHA-256.

`configs/low_latency.yaml` is a 192×192 variant for CPUs that miss the 1 s
latency target at 256×256. Read its header first: it keeps AUROC at 1.0 on
the synthetic set but lost small-defect localization in testing, so validate
it on your own parts before using it in production.

## Tests

```bash
python -m pytest -q
```

The suite covers PaDiM against a NumPy reference, k-fold calibration,
profile versioning/rollback/mismatch guards, offline operation, determinism,
decision boundaries, preprocessing/quality, and a synthetic end-to-end
train → inspect run with AUROC and localization checks.

## Measured results

**Only real runs are reported here.** MVTec AD was not available on the
development machine, so no MVTec AUROC/FAR/FRR numbers are quoted. Run
`python main.py validate --category bottle` (then `--category all`) once the
dataset is at `datasets/mvtec_ad/`.

Environment: Intel Celeron N4500 (2 cores, no GPU), 7 GB RAM, Python 3.12,
torch 2.15.0+cpu. Background load varied from ~3 to ~14 on 2 cores during
these runs, so latency is pessimistic and training times fluctuate.

| Run | Result |
|---|---|
| Synthetic end-to-end, 256 px (30 good + 60 aug, k=3) | AUROC **1.0000**, good mean 4.03 vs defect mean 38.67, **15/15** defects flagged, **0/10** unseen good marked DEFECTIVE, localization ≤ 24 px, heatmap written, train **43.0 s** |
| Fresh-start demo, generated set (30 good + 120 aug, k=5) | AUROC **1.0000**, good mean 4.40 vs defect mean 37.55, **15/15** defects flagged, **0/10** unseen good marked DEFECTIVE, train **72.8 s**, steady latency 557–632 ms mean / 708–924 ms P95 |
| IP-camera run via local mock `/shot.jpg` (30 good + 10 good/15 defect test) | AUROC **1.0000**, good mean 4.64 vs defect mean 37.61, **15/15** defects flagged, **0/10** good marked DEFECTIVE, train **57.6 s**, latency ~241–259 ms mean |
| Test suite | **63 passed** (50 s latest; 63–165 s observed; load-dependent) |
| Latency, 256 px | 883 ms mean / 1214 ms p95 (fails the 1 s gate on this loaded Celeron) |
| Latency, 192 px | 517 ms mean / 645 ms p95, but the 192 E2E failed 24 px localization on one scratch (36.4 px) — not accuracy-equivalent |
| M2 preprocessing only, 256 px | 4.5 ms mean / 9.1 ms p95 |
| M2 preprocessing only, 1080p | 62.3 ms mean / 72.6 ms p95 |
| MVTec AD `bottle` | not run (dataset absent) |
| MVTec AD all 15 | not run (dataset absent) |

See `results/PHASE1_REPORT.md` for the full report and `NOTES.md` for
decisions, deviations and limitations.

## Limitations

- **Pose:** PaDiM is position-specific; use a fixed camera/jig. Varying pose
  will raise good-image scores.
- **Small defects:** at 256×256 a defect smaller than one feature cell
  (~8 px) can vanish. 320×320 / ROI crops / tiling are later experiments.
- **Logical defects** (missing or misplaced components) are harder than
  surface defects; Phase 1 targets structural/appearance anomalies.
- **Lighting drift** is the most common cause of false rejects.
- Thresholds are mildly conservative because the final fit sees all data.

## Layout

```
main.py                     CLI: joins all modules
configs/default.yaml        all Phase-1 settings
module_0_common/            typed contracts, config, seeding, timing
module_1_acquisition/       file/folder loader + laptop/USB webcam capture
module_2_preprocessing/     quality, ROI, optional alignment/photometry, letterbox
module_3_learning/          backbone, embedding, PaDiM, trainer, profile store
module_4_inspection/        anomaly map, heatmap, inspector
module_5_decision/          k-fold calibration, GOOD/REVIEW/DEFECTIVE
module_6_evaluation/        MVTec adapter, metrics, synthetic demo parts
scripts/make_demo_dataset.py deterministic synthetic demo set
scripts/summarize_inspect.py AUROC/status/latency from inspect --jsonl
scripts/fetch_weights.py    one-time weights download (only network user)
scripts/run_validation.py   MVTec benchmark → results/
tests/                      unit + synthetic end-to-end + guards
DEMO_GUIDE.md               fresh-start synthetic and webcam walkthrough
MODULES.md                  module map, dependency rules and model choices
```
