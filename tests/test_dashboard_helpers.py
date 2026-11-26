"""Tests for the dashboard server's camera-URL handling.

The intro page sends whatever the operator typed; these helpers make it safe
before the URL reaches OpenCV or urllib.
"""

from __future__ import annotations

import pytest

from dashboard_server import _is_snapshot_url, _normalize_camera_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # bare phone IP: assume the IP Webcam defaults
        ("192.168.1.100", "http://192.168.1.100:8080/video"),
        # host + port, no path
        ("192.168.1.100:8080", "http://192.168.1.100:8080/video"),
        ("http://192.168.1.100", "http://192.168.1.100:8080/video"),
        ("http://192.168.1.100:8080", "http://192.168.1.100:8080/video"),
        ("http://192.168.1.100:8080/", "http://192.168.1.100:8080/video"),
        # explicit endpoints are never rewritten
        ("http://192.168.1.100:8080/video", "http://192.168.1.100:8080/video"),
        ("http://192.168.1.100:8080/shot.jpg", "http://192.168.1.100:8080/shot.jpg"),
        ("https://camera.local/stream", "https://camera.local/stream"),
        ("rtsp://192.168.1.100:554/stream", "rtsp://192.168.1.100:554/stream"),
    ],
)
def test_normalize_camera_url(raw: str, expected: str) -> None:
    assert _normalize_camera_url(raw) == expected


def test_normalize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _normalize_camera_url("   ")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://phone:8080/shot.jpg", True),
        ("http://phone:8080/photo.jpeg", True),
        ("http://phone:8080/frame.png", True),
        ("http://phone:8080/shot", True),
        ("http://phone:8080/video", False),
        ("http://phone:8080/stream.mjpg", False),
        ("rtsp://phone:554/stream", False),
    ],
)
def test_is_snapshot_url(url: str, expected: bool) -> None:
    assert _is_snapshot_url(url) is expected
