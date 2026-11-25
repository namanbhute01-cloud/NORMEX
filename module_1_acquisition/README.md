# Module 1 — Acquisition

Turns image files, folders, a laptop/USB webcam or a mobile IP camera into
`Frame` objects. No AI. The only network access is the camera URL you pass;
there is no cloud service.

| File | Purpose |
|---|---|
| `loader.py` | `list_images(dir)`, `load_frame(path)`, `IMAGE_EXTS`; deterministic ordering |
| `camera.py` | `Camera`, `SnapshotCamera`, `resolve_url()`, `redact_url()`, `capture_session()`, `save_frame()` |
| `scripts/mock_ipcam.py` | Local `/video` + `/shot.jpg` server for testing without a phone |

## File / folder flow

```python
from module_1_acquisition.loader import list_images, load_frame
```

## Mobile IP camera flow

Most phone apps (Android **IP Webcam**, and iOS equivalents) expose:

```
http://PHONE_IP:8080/video      MJPEG stream
http://PHONE_IP:8080/shot.jpg   single JPEG
```

Snapshot polling is recommended: every frame is a fresh HTTP request, so a
slow network cannot build a queue of stale frames.

```bash
# phone at 192.168.1.23, default IP Webcam port
python main.py capture \
  --ip 192.168.1.23 --port 8080 --snapshot \
  --out datasets/ipcam/good --count 30 --interval 0.5

# or an explicit stream URL
python main.py capture \
  --url http://192.168.1.23:8080/video \
  --out datasets/ipcam/good --count 30 --interval 0.5

# or an explicit snapshot URL
python main.py capture \
  --snapshot-url http://192.168.1.23:8080/shot.jpg \
  --out datasets/ipcam/good --count 30 --interval 0.5
```

`--ip` builds `http://IP:PORT/video`; adding `--snapshot` changes it to
`/shot.jpg`. Use `--url` / `--snapshot-url` when the app uses a different path
or port.

If the app has a login, put the credentials in the URL:

```
http://user:password@192.168.1.23:8080/shot.jpg
```

`SnapshotCamera` turns that into an HTTP Basic header and removes the
credentials from the URL before any log line. Stream URLs with embedded
credentials are passed to OpenCV as-is; snapshot mode is the reliable choice
for authenticated phone apps.

A snapshot request has a timeout (`--timeout`, default 5 s) and a 32 MB size
cap. A failed request raises `CameraError`; the CLI exits with code 2.

For stream mode, `Camera` reads back the actual width, height, FPS, backend
and exposure after warm-up. A phone that ignores a resolution request is
reported truthfully, and training uses the delivered frame size.

## Laptop webcam flow

```bash
python main.py capture --list
python main.py capture --out datasets/camera/good --count 30 --interval 0.5
```

`Camera.open()` discards warm-up frames so auto-exposure and
auto-white-balance settle before the first saved frame. `save_frame()` never
overwrites, so repeated runs accumulate `good_000.png`, `good_001.png`, ...
safely.

`--preview` needs an OpenCV build with GUI support. With
`opencv-python-headless` it prints a warning and continues headless; capture
still works.

## Testing without a phone

```bash
python scripts/mock_ipcam.py --host 127.0.0.1 --port 8099
python main.py capture \
  --ip 127.0.0.1 --port 8099 --snapshot \
  --out datasets/ipcam/good --count 3 --interval 0 --countdown 0 --warmup 0
```

The mock serves the same MJPEG and JPEG endpoints as a phone, so the HTTP
code path is real. It does not test phone optics, focus or compression.

## Acquisition discipline

The camera is a fixed acquisition device: keep the part, lighting and camera
still, and capture the same view that inspection will see. Software alignment
is available in M2 but off by default; a fixed jig is the primary control.

Vendor SDKs (Basler `pypylon`, GenICam `harvesters`) can be added behind the
same `Camera` interface later.
