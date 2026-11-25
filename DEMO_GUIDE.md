# Demo guide — from a fresh start

Three flows:

1. **No camera:** generate a synthetic part set, train and inspect.
2. **Mobile IP camera:** use a phone as the camera over local Wi-Fi.
3. **Laptop webcam:** fallback if there is no phone.

Everything runs offline on CPU. With the IP camera, frames travel only over
your local Wi-Fi; no cloud service is involved.

---

## 0. Fresh install

```bash
cd /home/naman/project/main_folder

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU-only PyTorch first (one time, needs internet)
pip install torch==2.15.0+cpu torchvision==0.29.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# One-time ResNet18 weights fetch (the only other network step)
python scripts/fetch_weights.py
```

Verify:

```bash
python -c "import cv2, torch; print(cv2.__version__, torch.__version__)"
python main.py --help
```

Expected commands: `train`, `capture`, `inspect`, `list`, `rollback`,
`validate`.

If `.venv/` and `pretrained/resnet18.pth` already exist, skip to step 1.

---

## 1. Synthetic demo (no camera)

```bash
python scripts/make_demo_dataset.py --out datasets/demo
```

```
datasets/demo/
├── good/       30 good parts      <- training only
├── new_good/   10 unseen good     <- test
├── defect/     15 defective       <- test
└── masks/      15 defect masks    <- evaluation only
```

Train on good images only:

```bash
python main.py train \
  --product demo \
  --images datasets/demo/good \
  --config configs/default.yaml
```

Inspect unseen good and defective parts:

```bash
python main.py inspect \
  --product demo \
  --image datasets/demo/new_good \
  --out results/demo/good \
  --jsonl results/demo/good.jsonl

python main.py inspect \
  --product demo \
  --image datasets/demo/defect \
  --out results/demo/defect \
  --jsonl results/demo/defect.jsonl
```

Summarize:

```bash
python scripts/summarize_inspect.py \
  --good results/demo/good.jsonl \
  --defect results/demo/defect.jsonl
```

Heatmaps are written at the **original photo resolution**, e.g.
`results/demo/defect/defect_000_scratch_heatmap.png`.

This set is deliberately easy. It is a plumbing demo, not a benchmark.

---

## 2. Mobile IP camera demo

### 2.1 Prepare the phone

1. Put the phone and laptop on the **same Wi-Fi**. Guest networks often block
   device-to-device traffic; use the normal network.
2. Install an IP-camera app:
   - Android: **IP Webcam** (Pavel Khlebovich) is the common choice.
   - iPhone: any app that serves MJPEG or JPEG snapshots over HTTP.
3. Open the app and start the server. It shows an address such as:

   ```
   http://192.168.1.23:8080
   ```

   Most Android apps expose:

   ```
   http://192.168.1.23:8080/video      MJPEG stream
   http://192.168.1.23:8080/shot.jpg   single JPEG
   ```

   Use whatever URL the app actually shows; the port is not always 8080.

### 2.2 Verify from the laptop

```bash
curl -o /tmp/phone.jpg http://192.168.1.23:8080/shot.jpg
python -c "import cv2; im=cv2.imread('/tmp/phone.jpg'); print(im.shape)"
```

The shape should be the phone camera resolution. If this fails, fix the
phone/Wi-Fi/firewall before touching the model.

If the app has a login, put the credentials in the URL:

```bash
curl -o /tmp/phone.jpg http://user:password@192.168.1.23:8080/shot.jpg
```

Passwords in URLs are redacted in the program's log output.

### 2.3 Capture good parts

Mount the phone so the part fills most of the frame. Keep the phone, part and
lighting fixed. `--snapshot` is recommended: every captured frame is a fresh
request, so a slow network cannot queue stale frames.

```bash
python main.py capture \
  --ip 192.168.1.23 \
  --port 8080 \
  --snapshot \
  --out datasets/ipcam/good \
  --count 30 \
  --interval 0.5 \
  --countdown 3
```

This writes `good_000.png` … `good_029.png`. Move the part slightly between
shots so the model sees normal pose variation, but **never show a defect
during training**.

If your app only provides an MJPEG stream, use the URL directly:

```bash
python main.py capture \
  --url http://192.168.1.23:8080/video \
  --out datasets/ipcam/good \
  --count 30 \
  --interval 0.5
```

For a non-default snapshot path:

```bash
python main.py capture \
  --snapshot-url http://192.168.1.23:8080/shot.jpg \
  --out datasets/ipcam/good \
  --count 30 \
  --interval 0.5
```

### 2.4 Train on the captured good parts

```bash
python main.py train \
  --product ipcam_part \
  --images datasets/ipcam/good \
  --config configs/default.yaml
```

You should see `good images: 30 (+120 augmented)` and
`[train] saved ipcam_part/v001`. Defect images are not read.

### 2.5 Capture test parts

```bash
python main.py capture \
  --ip 192.168.1.23 --port 8080 --snapshot \
  --out datasets/ipcam/new_good \
  --prefix new_good \
  --count 10 --interval 0.5

python main.py capture \
  --ip 192.168.1.23 --port 8080 --snapshot \
  --out datasets/ipcam/defect \
  --prefix defect \
  --count 15 --interval 0.5
```

Show only good parts for the first set and only damaged parts for the second.
Keep the same phone position and lighting as training.

### 2.6 Inspect

```bash
python main.py inspect \
  --product ipcam_part \
  --image datasets/ipcam/new_good \
  --out results/ipcam/good \
  --jsonl results/ipcam/good.jsonl

python main.py inspect \
  --product ipcam_part \
  --image datasets/ipcam/defect \
  --out results/ipcam/defect \
  --jsonl results/ipcam/defect.jsonl

python scripts/summarize_inspect.py \
  --good results/ipcam/good.jsonl \
  --defect results/ipcam/defect.jsonl
```

Open the heatmaps in `results/ipcam/`. A real defect should be hot and
localized; a good part should stay cool. `BAD_IMAGE` means the photo failed
the quality/alignment gate, not that the part is defective.

### 2.7 No phone? Use the mock IP camera

The repository includes a local mock that serves the same `/video` and
`/shot.jpg` endpoints:

```bash
# terminal 1
python scripts/mock_ipcam.py --host 127.0.0.1 --port 8099

# terminal 2 — snapshot mode
python main.py capture \
  --ip 127.0.0.1 --port 8099 --snapshot \
  --out datasets/ipcam/good \
  --count 30 --interval 0 --countdown 0 --warmup 0

# or stream mode
python main.py capture \
  --url http://127.0.0.1:8099/good/video \
  --out datasets/ipcam/stream \
  --count 3 --interval 0 --countdown 0 --warmup 3
```

This exercises the real HTTP/JPEG/MJPEG code path. It does **not** test phone
optics, focus, compression or lighting.

---

## 3. Laptop webcam fallback

```bash
python main.py capture --list
python main.py capture --out datasets/camera/good --count 30 --interval 0.5
python main.py train --product camera_part --images datasets/camera/good
python main.py capture --out datasets/camera/test --count 5 --prefix test
python main.py inspect --product camera_part --image datasets/camera/test
```

`--preview` needs a GUI OpenCV build. With `opencv-python-headless` it prints
a warning and captures without a window.

The laptop webcam is a fallback: it is usually lower quality and harder to
mount rigidly than a phone on a simple stand.

---

## 4. Measured clean IP-camera run

This is a real run from this repository, using the mock phone serving the
synthetic set over HTTP snapshots. It validates acquisition + learning +
inspection, not real phone optics.

| Item | Result |
|---|---:|
| Training good frames captured over HTTP | 30 |
| Training time | **57.6 s** |
| `t_lo` / `t_hi` | 6.444 / 7.411 |
| Unseen good / defect test frames | 10 / 15 |
| Image AUROC | **1.0000** |
| Good mean score | 4.644 |
| Defect mean score | 37.608 |
| Defects flagged | **15/15** |
| Good marked DEFECTIVE | **0/10** |
| Good latency | 240.6 ms mean / 293.3 ms P95 |
| Defect latency | 258.6 ms mean / 342.3 ms P95 |
| Preprocessing only | ~6–7 ms mean |

Machine: Intel Celeron N4500, 2 cores, no GPU. Load varies, so re-measure on
the target laptop.

Artifacts:

- `datasets/ipcam/` — frames captured through HTTP
- `results/ipcam/` — heatmaps, JSONL, `summary.txt`
- `profiles/ipcam_demo/v001` — trained profile

---

## 5. Re-running

```bash
python main.py list
python main.py rollback --product ipcam_part

# Start over
rm -rf profiles/ipcam_part results/ipcam datasets/ipcam
```

Changing `configs/default.yaml` changes the feature contract. Existing
profiles refuse to load; retrain them.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `snapshot request failed` | Phone and laptop on different Wi-Fi, app stopped, wrong IP/port, or firewall/VPN blocking |
| `could not open IP camera stream` | Try `--snapshot` with `/shot.jpg`; some apps do not serve MJPEG reliably |
| Snapshot is slow | Lower phone resolution; use 5 GHz Wi-Fi; keep the phone close to the router |
| Image is rotated | Rotate the phone/app to the same orientation for training and inspection |
| Credentials rejected | Use `http://user:pass@IP:PORT/shot.jpg`; snapshot mode supports Basic auth |
| `no usable local cameras found` | Only affects `--list`; IP cameras use `--ip`/`--url` and do not need a local index |
| All images `BAD_IMAGE` | Too dark/blurry/empty/wrong size; recapture with better light and framing |
| `profile ... config_hash` mismatch | Config changed after training; retrain the product |
| Training is slow | Expected on a 2-core CPU. Use `configs/low_latency.yaml` only after checking localization |
| `NNPACK ... Unsupported hardware` | Harmless; PyTorch falls back to CPU kernels |

## 7. What this demo does not prove

- MVTec AD accuracy (dataset absent; `validate` is not run)
- real production accuracy or defect coverage
- phone optics, focus, compression or lighting quality
- the < 1 s latency gate on the target laptop
- that a phone image is sharp enough for the smallest scratches
