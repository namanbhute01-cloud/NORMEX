#!/usr/bin/env python3
"""NORMEX live dashboard server — stdlib only, offline, CPU-friendly.

Serves the dashboard UI plus a small JSON API:

    GET /                       dashboard.html
    GET /api/status             datasets, profiles, latest benchmark, system
    GET /api/results?limit=50   recent verdict rows from results/**/*.jsonl
    GET /api/camera/frame       one live JPEG from the configured camera
    GET /api/inspect/live       capture + inspect one frame (product=...)
    GET /api/heatmap?p=...      heatmap PNG under results/

The camera URL comes from (in order): --camera-url / --snapshot-url,
$NORMEX_CAMERA_URL, or configs/camera.json {"url": ...} / {"snapshot_url": ...}.
It is never taken from a query string, so the server cannot be used as a proxy.

Run:
    python dashboard_server.py --camera-url http://192.168.1.100:8080/video --open
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module_0_common.config import config_hash, load_config, resolve_path  # noqa: E402
from module_0_common.types import Frame  # noqa: E402
from module_1_acquisition.camera import SnapshotCamera, redact_url  # noqa: E402
from module_2_preprocessing.pipeline import preprocess  # noqa: E402
from module_2_preprocessing.transform import map_to_source  # noqa: E402
from module_3_learning.backbone import Backbone  # noqa: E402
from module_3_learning.profile_store import (  # noqa: E402
    ProfileError,
    ProfileStore,
)
from module_4_inspection.anomaly_map import overlay  # noqa: E402
from module_4_inspection.inspector import Inspector  # noqa: E402
from module_5_decision.decide import decide  # noqa: E402

RESULTS_DIR = ROOT / "results"
DATASETS_DIR = ROOT / "datasets"
LIVE_DIR = RESULTS_DIR / "live"
MAX_JSONL_LINES = 2000

_CONFIG_CANDIDATES = (
    "configs/mvtec_192.yaml",
    "configs/mvtec_192_exposure.yaml",
    "configs/default.yaml",
    "configs/mvtec_fast.yaml",
    "configs/low_latency.yaml",
    "configs/mvtec_hires.yaml",
)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(self, camera_url: str, snapshot_url: str, config_path: str | None):
        self.camera_url = camera_url
        self.snapshot_url = snapshot_url
        self.config_path = config_path
        self.store = ProfileStore(ROOT / "profiles")
        self._cache: dict[str, tuple[Backbone, object, Inspector, dict]] = {}
        self._lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._inspect_lock = threading.Lock()
        # Live MJPEG streams keep one capture open: re-opening per request
        # costs ~0.5-1.5 s on a phone camera and blows the latency budget.
        self._capture: cv2.VideoCapture | None = None
        self._capture_url = ""
        self._capture_lock = threading.Lock()

    # -- camera configuration (set from the Live Camera tab) --------------------- #
    def set_camera(self, url: str) -> None:
        """Point the server at a new camera and persist the choice.

        Image URLs (``.jpg/.jpeg/.png/.bmp`` or ``/shot``) are treated as
        snapshot endpoints; everything else is opened as a stream.
        """

        url = _normalize_camera_url(url)
        with self._camera_lock:
            if _is_snapshot_url(url):
                self.snapshot_url = url
                self.camera_url = ""
            else:
                self.camera_url = url
                self.snapshot_url = ""
        self._reset_capture()
        _persist_camera(url)

    def _reset_capture(self) -> None:
        with self._capture_lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except cv2.error:
                    pass
            self._capture = None
            self._capture_url = ""

    def _current_camera(self) -> tuple[str, str]:
        with self._camera_lock:
            return self.camera_url, self.snapshot_url

    # -- config / profile helpers ------------------------------------------ #
    def _configs_to_try(self) -> list[str]:
        if self.config_path:
            return [self.config_path]
        return [str(ROOT / c) for c in _CONFIG_CANDIDATES]

    def resolve_model(self, product_id: str):
        """Return (backbone, profile, inspector, cfg) trying known configs."""
        with self._lock:
            if product_id in self._cache:
                return self._cache[product_id]

            last_error: Exception | None = None
            for candidate in self._configs_to_try():
                try:
                    cfg = load_config(candidate)
                    backbone = Backbone(resolve_path(cfg["backbone"]["weights"]))
                    profile = self.store.load(
                        product_id,
                        expected_config_hash=config_hash(cfg),
                        expected_backbone_sha256=backbone.sha256,
                    )
                except (ProfileError, FileNotFoundError, ValueError) as exc:
                    last_error = exc
                    continue
                inspector = Inspector(backbone, profile, cfg)
                self._cache[product_id] = (backbone, profile, inspector, cfg)
                return self._cache[product_id]
            raise ProfileError(f"cannot load profile '{product_id}': {last_error}")

    # -- camera ------------------------------------------------------------ #
    def camera_info(self) -> dict:
        camera_url, snapshot_url = self._current_camera()
        snapshot = snapshot_url or (camera_url if _is_snapshot_url(camera_url) else "")
        stream = "" if snapshot else camera_url
        return {
            "configured": bool(stream or snapshot),
            "stream_url": redact_url(stream) if stream else "",
            "snapshot_url": redact_url(snapshot) if snapshot else "",
        }

    def grab_frame(self, timeout: float = 5.0) -> Frame:
        camera_url, snapshot_url = self._current_camera()
        snapshot = snapshot_url or (camera_url if _is_snapshot_url(camera_url) else "")
        if snapshot:
            cam = SnapshotCamera(snapshot, timeout=timeout, warmup=0)
            frame = cam._fetch()
            return Frame(image=_shrink_live_frame(frame.image), source=frame.source)
        if not camera_url:
            raise RuntimeError(
                "no camera configured; start the server with "
                "--camera-url http://PHONE_IP:8080/video or connect from the Live Camera tab"
            )
        with self._capture_lock:
            if self._capture is None or self._capture_url != camera_url:
                if self._capture is not None:
                    self._capture.release()
                cap = cv2.VideoCapture(camera_url, cv2.CAP_ANY)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"cannot open camera stream {redact_url(camera_url)}")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._capture = cap
                self._capture_url = camera_url
            # Keep the newest frame; MJPEG backends can queue stale frames.
            image = None
            for _ in range(2):
                ok, frame_img = self._capture.read()
                if not ok or frame_img is None:
                    self._reset_capture_locked()
                    raise RuntimeError("camera returned no frame")
                image = frame_img
        return Frame(image=_shrink_live_frame(image), source=redact_url(camera_url))

    def _reset_capture_locked(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except cv2.error:
                pass
        self._capture = None
        self._capture_url = ""

    # -- status ------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "ok": True,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": self.camera_info(),
            "datasets": _scan_datasets(),
            "profiles": _scan_profiles(),
            "benchmark": _latest_benchmark(),
            "system": _system_info(),
        }

    def inspect_live(self, product_id: str) -> dict:
        # One inference at a time: the target CPU has two cores and the 3 s
        # auto-inspect timer can overlap with a manual click.
        with self._inspect_lock:
            return self._inspect_live(product_id)

    def _inspect_live(self, product_id: str) -> dict:
        started = time.perf_counter()
        backbone, profile, inspector, cfg = self.resolve_model(product_id)
        frame = self.grab_frame()
        # Live camera frames arrive at the sensor's native resolution and are
        # often framed wider than the close-up training images.  Keep the
        # photometric/blur gates, but do not reject a frame merely for its
        # pixel dimensions or for a smaller foreground fraction.
        quality_limits = dict(profile.preprocess.get("quality_limits") or {})
        if quality_limits:
            quality_limits["width"] = 0
            quality_limits["height"] = 0
            if quality_limits.get("fg_min") is not None:
                quality_limits["fg_min"] = min(float(quality_limits["fg_min"]), 0.01)
        prepared = preprocess(
            frame,
            cfg,
            stored=profile.preprocess,
            quality_limits=quality_limits or None,
        )
        inspection = inspector.inspect(prepared) if prepared.quality_ok else None
        verdict = decide(
            prepared,
            inspection,
            profile.calib,
            product_id,
            profile.version,
            (time.perf_counter() - started) * 1000.0,
        )
        verdict.preprocess_ms = prepared.latency_ms
        payload = {
            "product_id": verdict.product_id,
            "status": verdict.status,
            "score": verdict.score,
            "score_norm": verdict.score_norm,
            "t_lo": verdict.t_lo,
            "t_hi": verdict.t_hi,
            "latency_ms": verdict.latency_ms,
            "preprocess_ms": verdict.preprocess_ms,
            "profile_version": verdict.profile_version,
            "message": verdict.message,
            "heatmap_url": "",
            "ts": time.time(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if inspection is not None:
            source_map = map_to_source(inspection.anomaly_map, prepared.geometry)
            roi_x, roi_y, roi_w, roi_h = prepared.geometry["roi"]
            mask = np.zeros(source_map.shape, dtype=np.uint8)
            mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = 255
            heatmap = overlay(
                frame.image,
                source_map,
                vmin=0.0,
                vmax=profile.calib.t_hi * 1.5,
                mask=mask,
            )
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            out = LIVE_DIR / f"{product_id}_live_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(out), heatmap)
            rel_path = str(out.relative_to(ROOT))
            payload["heatmap_url"] = f"/api/heatmap?p={urllib.parse.quote(rel_path)}"
            payload["heatmap_path"] = rel_path
        _append_live_result(payload)
        return payload


# --------------------------------------------------------------------------- #
# disk scanners
# --------------------------------------------------------------------------- #
def _count_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _scan_datasets() -> list[dict]:
    rows: list[dict] = []
    if not DATASETS_DIR.is_dir():
        return rows
    for category_dir in sorted(p for p in DATASETS_DIR.iterdir() if p.is_dir()):
        if category_dir.name.startswith("."):
            continue
        for cat in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            train_good = _count_images(cat / "train" / "good")
            if train_good == 0 and (cat / "train" / "images").is_dir():
                train_good = _count_images(cat / "train" / "images")
            rows.append(
                {
                    "root": category_dir.name,
                    "name": cat.name,
                    "train_good": train_good,
                    "test_good": _count_images(cat / "test" / "good"),
                    "test_defect": sum(
                        _count_images(sub)
                        for sub in (cat / "test").iterdir()
                        if sub.is_dir() and sub.name != "good"
                    )
                    if (cat / "test").is_dir()
                    else 0,
                }
            )
    return rows


def _scan_profiles() -> list[dict]:
    rows: list[dict] = []
    root = ROOT / "profiles"
    if not root.is_dir():
        return rows
    for product_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        current_file = product_dir / "current"
        current = current_file.read_text(encoding="utf-8").strip() if current_file.is_file() else ""
        versions = sorted(p.name for p in product_dir.iterdir() if p.is_dir())
        report = {}
        if current:
            report_file = product_dir / current / "train_report.json"
            if report_file.is_file():
                try:
                    report = json.loads(report_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    report = {}
        rows.append(
            {
                "product_id": product_dir.name,
                "current": current,
                "versions": versions,
                "n_good": report.get("n_good"),
                "train_seconds": report.get("total_seconds"),
            }
        )
    return rows


def _latest_benchmark() -> dict:
    files = sorted(RESULTS_DIR.glob("mvtec_*.json")) if RESULTS_DIR.is_dir() else []
    if not files:
        return {"available": False}
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False}
    cats = payload.get("categories", [])
    return {
        "available": True,
        "file": files[-1].name,
        "timestamp": payload.get("timestamp", ""),
        "all_gates_pass": payload.get("all_gates_pass", False),
        "categories": cats,
        "environment": payload.get("environment", {}),
    }


def _system_info() -> dict:
    info: dict = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import psutil

        info["cpu_percent"] = psutil.cpu_percent(interval=None)
        info["mem_used_pct"] = psutil.virtual_memory().percent
        info["n_threads"] = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001 - psutil is optional at runtime
        pass
    return info


def _read_results(limit: int) -> list[dict]:
    rows: list[dict] = []
    files = []
    if RESULTS_DIR.is_dir():
        files = sorted(
            RESULTS_DIR.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )[-6:]
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-MAX_JSONL_LINES:]
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_source"] = str(path.relative_to(ROOT))
            rows.append(row)
    # newest first; rows in a JSONL file are appended chronologically
    rows.sort(key=lambda r: float(r.get("ts") or 0.0))
    return rows[-limit:][::-1]


def _append_live_result(payload: dict) -> None:
    """Append one live verdict to results/live/live.jsonl (best effort)."""

    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        target = LIVE_DIR / "live.jsonl"
        if target.is_file() and target.stat().st_size > 25 * 1024 * 1024:
            target.replace(LIVE_DIR / f"live_{int(time.time())}.jsonl")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


def _normalize_camera_url(url: str) -> str:
    """Accept ``192.168.1.100`` and expand it to an IP Webcam stream URL."""

    url = (url or "").strip()
    if not url:
        raise ValueError("camera URL must not be empty")
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme in ("http", "https") and parsed.path in ("", "/"):
        netloc = parsed.netloc
        if parsed.scheme == "http" and parsed.port is None:
            netloc += ":8080"
        url = urllib.parse.urlunsplit((parsed.scheme, netloc, "/video", "", ""))
    return url


def _is_snapshot_url(url: str) -> bool:
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".bmp")) or path.endswith("/shot")


def _shrink_live_frame(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Downscale a camera frame for quality checks and scoring.

    Phone cameras commonly deliver 1920x1080; the profile runs at 192-256 px,
    so keeping the full sensor frame only adds decode/preprocess latency.
    """

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _persist_camera(url: str) -> None:
    """Remember the camera URL so the next start uses it automatically."""

    cfg_dir = ROOT / "configs"
    target = cfg_dir / "camera.json"
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        data = {"url": url, "snapshot_url": ""}
        if _is_snapshot_url(url):
            data = {"url": "", "snapshot_url": url}
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    except OSError:
        pass


def _check_camera_reachable(url: str, timeout: float = 2.5) -> None:
    """Fail fast on a wrong IP/port instead of hanging inside OpenCV."""

    import socket

    parsed = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    host = parsed.hostname
    if not host:
        raise ValueError("camera URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        raise ValueError(
            f"cannot reach {host}:{port} ({exc}). Check the phone's IP, port and Wi-Fi."
        ) from exc


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    state: AppState  # set in main()
    server_version = "NORMEX/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        if os.environ.get("NORMEX_HTTP_LOG"):
            super().log_message(fmt, *args)

    # -- helpers ----------------------------------------------------------- #
    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._send_json({"error": f"not found: {path.name}"}, HTTPStatus.NOT_FOUND)
            return
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # -- routes ------------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path in ("/", "/index.html"):
                self._send_file(ROOT / "dashboard.html", "text/html; charset=utf-8")
            elif path in ("/healthz", "/api/health"):
                self._send_json({"ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            elif path == "/api/status":
                self._send_json(self.state.status())
            elif path == "/api/results":
                limit = min(500, max(1, int(query.get("limit", ["50"])[0])))
                self._send_json({"results": _read_results(limit)})
            elif path == "/api/camera/frame":
                frame = self.state.grab_frame()
                ok, buffer = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("could not encode camera frame")
                body = buffer.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/inspect/live":
                product = query.get("product", ["screw"])[0]
                self._send_json(self.state.inspect_live(product))
            elif path == "/api/heatmap":
                rel = Path(query.get("p", [""])[0])
                full = (ROOT / rel).resolve()
                allowed = RESULTS_DIR.resolve() in full.parents
                if not allowed or not full.is_file():
                    self._send_json({"error": "heatmap not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(full, "image/png")
            else:
                self._send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except ProfileError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        """Configure the camera from the Live Camera tab (loopback only)."""

        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/camera":
            self._send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            return
        client = self.client_address[0]
        if client not in ("127.0.0.1", "::1", "localhost"):
            self._send_json(
                {"error": "camera can only be configured from the machine running the server"},
                HTTPStatus.FORBIDDEN,
            )
            return
        try:
            length = min(8192, int(self.headers.get("Content-Length") or 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            url = str(data.get("url", "")).strip()
            if not url:
                raise ValueError("camera URL is required")
            url = _normalize_camera_url(url)
            _check_camera_reachable(url)
            self.state.set_camera(url)
            self._send_json({"ok": True, "camera": self.state.camera_info()})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)


def _load_camera_settings() -> tuple[str, str]:
    url = os.environ.get("NORMEX_CAMERA_URL", "")
    snapshot = os.environ.get("NORMEX_SNAPSHOT_URL", "")
    cfg_file = ROOT / "configs" / "camera.json"
    if cfg_file.is_file() and not url and not snapshot:
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            url = data.get("url", "")
            snapshot = data.get("snapshot_url", "")
        except (json.JSONDecodeError, OSError):
            pass
    return url, snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NORMEX live dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera-url", default="", help="IP camera stream URL (MJPEG/RTSP/HTTP)")
    parser.add_argument("--snapshot-url", default="", help="JPEG snapshot URL")
    parser.add_argument("--config", default=None, help="force one config yaml")
    parser.add_argument("--open", action="store_true", help="open the browser")
    args = parser.parse_args(argv)

    env_url, env_snapshot = _load_camera_settings()
    camera_url = args.camera_url or env_url
    snapshot_url = args.snapshot_url or env_snapshot

    Handler.state = AppState(camera_url, snapshot_url, args.config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] NORMEX live dashboard on {url}")
    print(f"[dashboard] camera: {redact_url(snapshot_url or camera_url) or 'not configured'}")
    print("[dashboard] press Ctrl+C to stop")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
