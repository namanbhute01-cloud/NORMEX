"""M1: laptop/USB webcam, RTSP/CCTV IP camera, and mobile IP camera capture.

OpenCV only; no AI and no cloud.  Four acquisition modes feed the same BGR
``Frame`` contract:

* local webcam / USB camera      -> ``Camera(index=0)``
* RTSP / RTSP over TCP or UDP   -> ``Camera(url="rtsp://host:port/stream")``
* IP camera MJPEG / HTTP stream -> ``Camera(url="http://host:port/video")``
* IP camera JPEG snapshot       -> ``SnapshotCamera("http://host:port/shot.jpg")``

The ``Camera`` class auto-detects the URL scheme and lets ``cv2.VideoCapture``
choose the appropriate backend (FFMPEG, GStreamer, or V4L2).  The
``SnapshotCamera`` always uses a fresh HTTP GET per frame, which is often more
reliable than an MJPEG stream on a phone because a slow network cannot build a
queue of stale frames.

URLs containing ``user:password@`` are redacted before they appear in any log
output.  In snapshot mode the credentials are turned into an HTTP Basic header
and then removed from the printed URL.
"""

from __future__ import annotations

import base64
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from module_0_common.types import Frame

MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024


class CameraError(RuntimeError):
    """The camera could not be opened or a frame could not be read."""


@dataclass
class CameraInfo:
    source: str
    backend: str
    width: int
    height: int
    fps: float
    exposure: Optional[float] = None
    index: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def redact_url(url: str) -> str:
    """Hide the password in a camera URL before logging it."""

    try:
        parts = urllib.parse.urlsplit(url)
        if parts.password is None:
            return url
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
        return urllib.parse.urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )
    except ValueError:
        return url


def detect_url_scheme(url: str) -> str:
    """Return the scheme part of a URL lower‑cased, or ``'unknown'``."""

    try:
        return urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:
        return "unknown"


def is_rtsp_url(url: str) -> bool:
    """True if *url* looks like an RTSP address."""

    return detect_url_scheme(url) in {"rtsp", "rtsps"}


def is_http_url(url: str) -> bool:
    """True if *url* is an HTTP or HTTPS address."""

    return detect_url_scheme(url) in {"http", "https"}


def resolve_url(ip: str, port: int = 8080, *, snapshot: bool = False) -> str:
    """Build an IP-camera URL from a phone's LAN address.

    Android "IP Webcam" and most iOS equivalents use ``/video`` for the MJPEG
    stream and ``/shot.jpg`` for a still.  The caller can override both with
    explicit ``--url`` / ``--snapshot-url`` arguments.

    The returned URL is always ``http://`` because most phone apps do not speak
    RTSP; use ``--url`` to point at a custom stream endpoint if needed.
    """

    host = str(ip).strip()
    if not host:
        raise ValueError("--ip must not be empty")
    if "://" in host:
        raise ValueError("--ip expects a host or IP, not a URL; use --url instead")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    path = "/shot.jpg" if snapshot else "/video"
    return f"http://{host}:{int(port)}{path}"


def _prepare_request(url: str) -> tuple[str, dict[str, str]]:
    """Strip userinfo from a URL and turn it into an HTTP Basic header."""

    parts = urllib.parse.urlsplit(url)
    headers = {"User-Agent": "CameraInspector/1.0"}
    if parts.username is None:
        return url, headers

    username = urllib.parse.unquote(parts.username)
    password = urllib.parse.unquote(parts.password or "")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    headers["Authorization"] = f"Basic {token}"

    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    clean = urllib.parse.urlunsplit(
        (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
    )
    return clean, headers


def _backend_name(capture: cv2.VideoCapture) -> str:
    getter = getattr(capture, "getBackendName", None)
    if getter is None:
        return "unknown"
    try:
        return str(getter())
    except cv2.error:
        return "unknown"


class Camera:
    """One OpenCV camera or network stream, opened once and reused.

    The *url* argument can be:

    * ``0`` or ``""``   – local webcam (index 0)
    * ``rtsp://host:port/stream``   – RTSP stream (requires FFMPEG or GStreamer)
    * ``http://host:port/video``   – MJPEG or HTTP stream
    * ``http://host:port/shot.jpg`` – JPEG snapshot URL (used by SnapshotCamera)

    OpenCV's ``VideoCapture`` auto-detects the backend; if the FFMPEG or
    GStreamer backend is available it will handle RTSP and most HTTP streams.
    """

    def __init__(
        self,
        index: int = 0,
        url: str = "",
        width: int = 1280,
        height: int = 720,
        warmup: int = 15,
    ) -> None:
        self.index: Optional[int] = None if url else int(index)
        self.url = str(url or "")
        self.width = int(width)
        self.height = int(height)
        self.warmup = max(0, int(warmup))
        self.capture: Optional[cv2.VideoCapture] = None
        self.info: Optional[CameraInfo] = None

    @property
    def source(self) -> str:
        return redact_url(self.url) if self.url else f"camera:{self.index}"

    def open(self) -> CameraInfo:
        # Choose the CV constant based on scheme – this hints OpenCV which
        # backend to initialise, but we always fall back to auto-detection
        # if the constant is not available or the open fails.
        capture = cv2.VideoCapture(self.url, cv2.CAP_ANY)

        if not capture.isOpened():
            # If CAP_ANY failed, try explicitly with FFMPEG (handles RTSP/RTSPs)
            if is_rtsp_url(self.url) or is_http_url(self.url):
                fallback = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if fallback.isOpened():
                    capture.release()
                    capture = fallback
                else:
                    # Try GStreamer fallback
                    try:
                        gst_cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
                        if gst_cap.isOpened():
                            capture.release()
                            capture = gst_cap
                    except Exception:
                        pass
            capture.release()
            if self.url:
                raise CameraError(
                    f"could not open camera stream {redact_url(self.url)}. "
                    "Check that the URL scheme is correct, the device is on the same "
                    "Wi‑Fi, the app is serving, and the port/path are correct.  "
                    "OpenCV may need the FFMPEG or GStreamer backend for RTSP streams."
                )
            raise CameraError(
                f"could not open camera index {self.index}. "
                "Close other camera apps, check permissions, or try --camera 1."
            )

        if self.url:
            # Reserve a single‑frame buffer so the newest frame is always
            # returned; some backends default to a deep queue which adds latency.
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass

        # Discard early frames: auto‑exposure settles, and network streams
        # often return a partial first frame.  At least one frame is always
        # read, even when warmup=0, to prove the source is live.
        last = None
        for _ in range(max(1, self.warmup)):
            ok, image = capture.read()
            if not ok or image is None:
                capture.release()
                raise CameraError(
                    f"{self.source} opened but returned no frames during warm-up"
                )
            last = image

        assert last is not None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or int(last.shape[1])
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or int(last.shape[0])
        try:
            exposure: Optional[float] = float(capture.get(cv2.CAP_PROP_EXPOSURE))
        except cv2.error:
            exposure = None

        info = CameraInfo(
            source=self.source,
            backend=_backend_name(capture),
            width=width,
            height=height,
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            exposure=exposure,
            index=self.index,
        )
        self.capture = capture
        self.info = info
        return info

    def read(self) -> Frame:
        if self.capture is None:
            raise CameraError("camera is not open; call open() first")
        ok, image = self.capture.read()
        if not ok or image is None:
            raise CameraError(f"{self.source} returned an empty frame")
        return Frame(image=image, source=self.source)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SnapshotCamera:
    """Poll a JPEG snapshot URL, one fresh HTTP request per frame."""

    def __init__(self, url: str, timeout: float = 5.0, warmup: int = 2) -> None:
        if not str(url).strip():
            raise ValueError("snapshot URL must not be empty")
        self.url = str(url)
        self.timeout = max(0.1, float(timeout))
        self.warmup = max(0, int(warmup))
        self.info: Optional[CameraInfo] = None

    @property
    def source(self) -> str:
        return redact_url(self.url)

    def _fetch(self) -> Frame:
        request_url, headers = _prepare_request(self.url)
        request = urllib.request.Request(request_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read(MAX_SNAPSHOT_BYTES)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise CameraError(
                f"snapshot request failed for {self.source}: {exc}. "
                "Check the phone and laptop are on the same Wi‑Fi and the app is "
                "serving the JPEG endpoint."
            ) from exc

        if not data:
            raise CameraError(f"snapshot URL {self.source} returned an empty body")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CameraError(
                f"snapshot URL {self.source} did not return a decodable JPEG/PNG "
                f"({len(data)} bytes)"
            )
        return Frame(image=image, source=self.source)

    def open(self) -> CameraInfo:
        # Warm‑up: a few fresh requests so the phone's HTTP server is ready.
        for _ in range(max(0, self.warmup)):
            self._fetch()
        self.info = CameraInfo(
            source=self.source,
            backend="http-snapshot",
            width=int(self._fetch().image.shape[1]),
            height=int(self._fetch().image.shape[0]),
            fps=0.0,
            exposure=None,
            index=None,
        )
        return self.info

    def read(self) -> Frame:
        if self.info is None:
            raise CameraError("snapshot camera is not open; call open() first")
        return self._fetch()

    def close(self) -> None:
        self.info = None

    def __enter__(self) -> "SnapshotCamera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_cameras(max_index: int = 5) -> list[int]:
    """Probe local camera indices 0..max_index-1 and return the usable ones.

    Only indices that return a valid frame via ``cv2.VideoCapture`` are included.
    """

    found: list[int] = []
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except (AttributeError, cv2.error):
        pass
    try:
        for index in range(max(1, int(max_index))):
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
            try:
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        found.append(index)
            finally:
                capture.release()
    finally:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
        except (AttributeError, cv2.error):
            pass
    return found


def _next_path(out_dir: Path, prefix: str) -> Path:
    index = 0
    while True:
        candidate = out_dir / f"{prefix}_{index:03d}.png"
        if not candidate.exists():
            return candidate
        index += 1


def save_frame(frame: Frame, out_dir: str | Path, prefix: str = "good") -> Path:
    """Write one BGR frame as PNG, never overwriting an existing file."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = _next_path(directory, prefix)
    if not cv2.imwrite(str(path), frame.image):
        raise CameraError(f"could not write captured frame: {path}")
    return path


def capture_session(
    out_dir: str | Path,
    *,
    index: int = 0,
    url: str = "",
    snapshot_url: str = "",
    count: int = 30,
    interval_s: float = 1.0,
    width: int = 1280,
    height: int = 720,
    warmup: int = 15,
    preview: bool = False,
    countdown_s: float = 3.0,
    prefix: str = "good",
    timeout: float = 5.0,
) -> list[Path]:
    """Capture ``count`` frames from a local camera, stream or snapshot URL.

    ``preview`` needs a GUI build of OpenCV.  With the headless wheel the
    function prints a warning and continues without a window.
    """

    if count < 1:
        raise ValueError("capture count must be at least 1")
    if interval_s < 0:
        raise ValueError("capture interval must be >= 0")
    if countdown_s < 0:
        raise ValueError("capture countdown must be >= 0")
    if url and snapshot_url:
        raise ValueError("provide either url or snapshot_url, not both")

    if snapshot_url:
        camera: Camera | SnapshotCamera = SnapshotCamera(
            snapshot_url, timeout=timeout, warmup=warmup
        )
    else:
        camera = Camera(index=index, url=url, width=width, height=height, warmup=warmup)

    with camera:
        info = camera.info
        assert info is not None
        print(
            f"[capture] {info.source}: {info.width}x{info.height} "
            f"@ {info.fps:.1f} fps ({info.backend})"
        )
        if countdown_s:
            print(f"[capture] starting in {countdown_s:.0f}s — position the part")
            time.sleep(countdown_s)

        preview_ok = bool(preview)
        if preview_ok and not hasattr(cv2, "imshow"):
            print("[capture] preview is unavailable in this OpenCV build; continuing headless")
            preview_ok = False

        paths: list[Path] = []
        for number in range(1, count + 1):
            frame = camera.read()
            if preview_ok:
                try:
                    cv2.imshow("Camera Inspector — capture (q to stop)", frame.image)
                    key = cv2.waitKey(max(1, int(interval_s * 1000))) & 0xFF
                    if key == ord("q"):
                        print("[capture] stopped by user")
                        break
                except cv2.error:
                    print(
                        "[capture] preview window is not available in this OpenCV build; "
                        "continuing headless"
                    )
                    preview_ok = False
            elif interval_s:
                time.sleep(interval_s)

            path = save_frame(frame, out_dir, prefix=prefix)
            paths.append(path)
            print(f"[capture] {number}/{count} -> {path}")

        if preview_ok:
            cv2.destroyAllWindows()
        return paths