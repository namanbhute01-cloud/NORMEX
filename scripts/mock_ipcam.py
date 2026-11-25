#!/usr/bin/env python3
"""Local mock IP camera for testing the acquisition path without a phone.

It serves the same two endpoints most phone apps expose:

    http://HOST:PORT/<set>/video      MJPEG stream
    http://HOST:PORT/<set>/shot.jpg   single JPEG

where ``<set>`` is ``good``, ``new_good`` or ``defect``.  This is a test tool,
not a production server: it cycles through the images in the chosen folder.
"""

from __future__ import annotations

import argparse
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS: dict[str, list[Path]] = {}
COUNTERS: dict[str, int] = {name: 0 for name in ("good", "new_good", "defect")}
LOCK = threading.Lock()
FPS = 10.0


def _next_image(folder: str) -> Path:
    images = DATASETS[folder]
    with LOCK:
        index = COUNTERS[folder] % len(images)
        COUNTERS[folder] = (COUNTERS[folder] + 1) % len(images)
    return images[index]


def _jpeg(path: Path) -> bytes:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read mock frame: {path}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"could not encode mock frame: {path}")
    return encoded.tobytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "MockIPCam/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mock-ipcam] {self.address_string()} {fmt % args}")

    def _send_404(self) -> None:
        body = b"unknown mock camera path\n"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parts = urllib.parse.urlsplit(self.path)
        path = parts.path

        if path == "/":
            rows = "".join(
                f"<li><a href='/{name}/video'>{name}/video</a> | "
                f"<a href='/{name}/shot.jpg'>{name}/shot.jpg</a></li>"
                for name in DATASETS
            )
            body = f"<html><body><h3>Mock IP camera</h3><ul>{rows}</ul></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        stripped = path.strip("/")
        if stripped in {"shot.jpg", "video"}:
            folder, endpoint = "good", stripped
        else:
            folder, _, endpoint = stripped.partition("/")

        if folder not in DATASETS or endpoint not in {"shot.jpg", "video"}:
            self._send_404()
            return

        if endpoint == "shot.jpg":
            self._serve_snapshot(folder)
        else:
            self._serve_stream(folder)

    def _serve_snapshot(self, folder: str) -> None:
        try:
            jpeg = _jpeg(_next_image(folder))
        except Exception as exc:  # pragma: no cover - defensive
            body = str(exc).encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(jpeg)
        self.close_connection = True

    def _serve_stream(self, folder: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.close_connection = True
        delay = 1.0 / max(0.5, float(FPS))
        try:
            while True:
                jpeg = _jpeg(_next_image(folder))
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv: list[str] | None = None) -> int:
    global FPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", default="datasets/demo/good")
    parser.add_argument("--new-good", default="datasets/demo/new_good")
    parser.add_argument("--defect", default="datasets/demo/defect")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; use 0.0.0.0 to let a phone reach this laptop")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    FPS = args.fps

    folders = {
        "good": Path(args.good),
        "new_good": Path(args.new_good),
        "defect": Path(args.defect),
    }
    for name, folder in folders.items():
        if not folder.is_dir():
            parser.error(f"--{name.replace('_', '-')} folder not found: {folder}")
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not images:
            parser.error(f"no images in {folder}")
        DATASETS[name] = images

    server = MockServer((args.host, args.port), Handler)
    print(f"[mock-ipcam] serving on http://{args.host}:{args.port}")
    for name in DATASETS:
        print(f"[mock-ipcam]   /{name}/video    /{name}/shot.jpg  ({len(DATASETS[name])} images)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-ipcam] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
