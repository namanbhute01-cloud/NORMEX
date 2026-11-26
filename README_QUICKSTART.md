# NORMEX — Quick Start (Linux · Windows · Docker)

NORMEX is a CPU-only, offline visual inspection system. It learns what a good
part looks like from **good images only**, then flags defects with a heatmap.

---

## 1. Install

### Linux / macOS
```bash
cd main_folder
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.15.0+cpu torchvision==0.29.2+cpu
.venv/bin/pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd main_folder
python -m venv .venv
.venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu torchvision==0.29.2+cpu
.venv\Scripts\pip install -r requirements.txt
```

---

## 2. Start the dashboard

```bash
./start.sh                # Linux / macOS
start.bat                 # Windows
```

This checks your datasets, then starts the live dashboard at
<http://127.0.0.1:8765> and opens your browser. Press `Ctrl+C` to stop.

The UI has two pages:

1. **Welcome** — an animated NORMEX intro (mint/green design, rotating lens
   mark). It advances to the dashboard automatically after 5.5 s; press Enter
   or click the bottom badge to skip sooner. Add `?skip=1&instant=1` to
   the URL to bypass it (useful for kiosks or screenshots).
2. **Dashboard** — the inspection console. If a camera is already configured,
   NORMEX auto-opens the **Live Camera** tab after a short 2.4 s intro. The
   brand in the header returns to the Overview tab. A light/dark theme switch
   sits at the top right.

### Dashboard tabs
| Tab | What it shows |
|---|---|
| **Overview** | Latest MVTec benchmark (AUROC, missed defects, false rejects, train time, P95), gates, system load |
| **Live Camera** | Live phone-camera feed, product selector, **Inspect Now**, verdict + heatmap, auto-inspect every 3 s |
| **Results** | Recent inspection verdicts from `results/**/*.jsonl` |
| **Datasets** | Installed datasets and trained profiles |

---

## 3. Connect a mobile IP camera

On the phone install **IP Webcam** (Android) or an equivalent MJPEG app and
start the server. Then either:

**A. Use the Live Camera tab (recommended)**
Start `./start.sh` / `start.bat`, wait for the welcome animation (or press
Enter), open **Live Camera**, type the phone address
(`192.168.1.100`) in the **Phone camera** field and press **Connect**. It builds
`http://192.168.1.100:8080/video`, verifies the port, saves
`configs/camera.json`, and starts the feed automatically.

**B. Pass the URL once**
```bash
./start.sh dashboard --camera-url http://192.168.1.100:8080/video
```
or use JPEG snapshots (more reliable on weak Wi-Fi):
```bash
./start.sh dashboard --ip 192.168.1.100 --port 8080 --snapshot-url http://192.168.1.100:8080/shot.jpg
```

**C. Save it manually**
```bash
cp configs/camera.json.example configs/camera.json
# edit configs/camera.json -> {"url": "http://192.168.1.100:8080/video"}
./start.sh
```

The phone and the laptop must be on the same Wi-Fi network.

### Capture good frames (for training)
```bash
./start.sh capture --out datasets/my_part/good --camera-url http://192.168.1.100:8080/video --count 30
```

---

## 4. Train a product

Good images only — put them in a folder and run:

```bash
.venv/bin/python main.py train --product my_part --images datasets/my_part/good
```

Train every dataset installed under `datasets/` that has `train/good`
(pills, bottle, nut, avm, and any MVTec categories):

```bash
./start.sh train-all                 # skips profiles that already exist
./start.sh train-all --force         # retrain everything
```

MVTec benchmark (good images from `datasets/mvtec_ad/screw/train/good`):

```bash
./start.sh benchmark
# == python main.py validate --category screw --config configs/mvtec_fast.yaml
```

Results are written to `results/mvtec_*.json/.csv` and appear on the dashboard
Overview tab automatically. `configs/mvtec_hires.yaml` is the paper-faithful
PaDiM grid (64×64 patches, no augmentation) for the accuracy run;
`configs/mvtec_fast.yaml` is the low-memory variant; `configs/low_latency.yaml`
is the 192×192 variant for the < 1 s latency gate.

---

## 5. Inspect images from the CLI

```bash
.venv/bin/python main.py inspect --product my_part --image path/to/test.jpg \
    --out results/my_part --jsonl results/my_part/run.jsonl
.venv/bin/python scripts/summarize_inspect.py --good results/my_part/good.jsonl \
    --defect results/my_part/defect.jsonl
```

---

## 6. Extra datasets (Kaggle)

Convert detection/plain-image datasets into the good-only layout:

```bash
.venv/bin/python scripts/convert_datasets.py \
    --pills ~/datasets/mvtec_ad/pills \
    --bottle "/tmp/bottle_extract/Plastic Bottle Image Dataset" \
    --folder nut=/tmp/nut_extract/autorenders_nut_fixed/nut_images \
    --folder avm=/tmp/avm_extract/AVM
```

Output: `datasets/extra/<name>/train/good` + `test/good`, then train as usual:

```bash
.venv/bin/python main.py train --product pills --images datasets/extra/pills/train/good
```

---

## 7. Latency check (< 1000 ms P95 gate)

```bash
.venv/bin/python scripts/bench_latency.py --product screw --config configs/mvtec_fast.yaml
.venv/bin/python scripts/bench_latency.py --product screw --config configs/low_latency.yaml --threads 1
```

Writes `results/latency_*.json`.

---

## 8. Docker / service deploy (production)

```bash
docker compose up -d --build
curl http://127.0.0.1:8765/healthz
```

Or run the image directly:

```bash
docker build -t normex:latest .
docker run --rm -p 8765:8765 \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/profiles:/app/profiles" \
  -v "$PWD/results:/app/results" \
  -e NORMEX_CAMERA_URL="http://192.168.1.100:8080/video" \
  normex:latest serve
```

Linux systemd and Windows Task Scheduler instructions are in
[`deploy/README.md`](deploy/README.md). The service has a `/healthz` endpoint,
restart-on-crash, and writes one JSON line per live inspection to
`results/live/live.jsonl`.

Train inside the image:
```bash
docker run --rm -v "$PWD/datasets:/app/datasets" -v "$PWD/profiles:/app/profiles" \
  normex:latest train --product screw --images datasets/mvtec_ad/screw/train/good
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `camera: not configured` on dashboard | Start with `--camera-url` or create `configs/camera.json` |
| Camera shows nothing | Phone and laptop on same Wi-Fi; try `--snapshot-url http://IP:8080/shot.jpg`; check the app port |
| `ProfileMismatchError` | The profile was trained with a different config; retrain with the same `--config` |
| Dashboard page opens but says offline | The page needs `dashboard_server.py`; use `./start.sh`, not a direct file open |
| Training killed (OOM) | Use `--config configs/mvtec_fast.yaml` (augment 1, batch 4) or a smaller input size |
| Slow inference | Try `configs/low_latency.yaml` (192×192) and `--threads 1`; verify localization on your smallest defect |

---

## Project layout (key files)

```
main.py                     CLI: capture / train / inspect / list / rollback / validate
launch.py                   cross-platform launcher (used by start.sh & start.bat)
dashboard_server.py         stdlib HTTP API + live camera + live inspection
dashboard.html              dashboard UI (Overview / Live / Results / Datasets)
module_1_acquisition/       webcam, IP stream, JPEG snapshot, RTSP
module_2_preprocessing/     quality gate, letterbox, optional ROI/align/denoise
module_3_learning/          ResNet18 backbone, PaDiM, profile store (versioned)
module_4_inspection/        anomaly map + heatmap
module_5_decision/          calibration + GOOD/REVIEW/DEFECTIVE/BAD_IMAGE
module_6_evaluation/        MVTec adapter + AUROC/FAR/FRR/P95 metrics
scripts/                    validation harness, conversion, latency, demo data
configs/                    default.yaml, mvtec_fast.yaml, low_latency.yaml
```
