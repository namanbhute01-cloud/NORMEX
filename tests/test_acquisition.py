"""M1 acquisition and synthetic demo tooling.

No real camera is touched here: the tests cover the file side, the
non-overwriting writer, URL handling and the deterministic dataset generator.
The snapshot test uses a local HTTP server, not a phone.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from module_0_common.types import Frame
from module_1_acquisition.camera import (
    SnapshotCamera,
    capture_session,
    redact_url,
    resolve_url,
    save_frame,
)
from module_6_evaluation.synthetic import generate_dataset


def test_save_frame_never_overwrites(tmp_path: Path) -> None:
    frame = Frame(image=np.zeros((8, 8, 3), dtype=np.uint8), source="synthetic")

    first = save_frame(frame, tmp_path, prefix="good")
    second = save_frame(frame, tmp_path, prefix="good")

    assert first.name == "good_000.png"
    assert second.name == "good_001.png"
    assert cv2.imread(str(first)).shape == (8, 8, 3)
    assert cv2.imread(str(second)).shape == (8, 8, 3)


def test_capture_session_validates_count_before_opening_a_camera(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        capture_session(tmp_path, count=0)


def test_ip_camera_url_helpers() -> None:
    assert resolve_url("192.168.1.5") == "http://192.168.1.5:8080/video"
    assert (
        resolve_url("192.168.1.5", 4747, snapshot=True)
        == "http://192.168.1.5:4747/shot.jpg"
    )
    assert resolve_url("::1") == "http://[::1]:8080/video"
    assert (
        redact_url("http://user:secret@cam.local:8080/video")
        == "http://user:***@cam.local:8080/video"
    )
    with pytest.raises(ValueError):
        resolve_url("http://192.168.1.5:8080/video")


def test_snapshot_camera_reads_over_http(tmp_path: Path) -> None:
    image = np.full((16, 24, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = encoded.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/shot.jpg"
        with SnapshotCamera(url, timeout=2.0, warmup=0) as camera:
            info = camera.info
            assert info is not None
            assert info.backend == "http-snapshot"
            assert (info.width, info.height) == (24, 16)
            frame = camera.read()
            assert frame.image.shape == (16, 24, 3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)


def test_generate_dataset_counts_and_shapes(tmp_path: Path) -> None:
    info = generate_dataset(
        tmp_path / "demo",
        n_good=5,
        n_new_good=3,
        n_defect=3,
        size=96,
        seed=11,
    )

    assert len(list(info["good"].glob("*.png"))) == 5
    assert len(list(info["new_good"].glob("*.png"))) == 3
    assert len(list(info["defect"].glob("*.png"))) == 3
    assert len(list(info["masks"].glob("*.png"))) == 3

    for path in list(info["good"].glob("*.png")) + list(info["defect"].glob("*.png")):
        assert cv2.imread(str(path)).shape == (96, 96, 3)
