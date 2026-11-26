#!/usr/bin/env python3
"""NORMEX cross-platform launcher (Linux / macOS / Windows).

    python launch.py                     # start the live dashboard + open browser
    python launch.py setup               # verify datasets, generate demo if missing
    python launch.py train-demo          # train the synthetic demo profile if missing
    python launch.py train-all           # train every dataset that has no profile
    python launch.py benchmark           # run the MVTec AD benchmark (screw)
    python launch.py capture --camera-url http://IP:8080/video --count 30

Camera URL resolution (first match wins):
    --camera-url / --snapshot-url
    $NORMEX_CAMERA_URL / $NORMEX_SNAPSHOT_URL
    configs/camera.json  {"url": "http://PHONE_IP:8080/video"}

The start.sh / start.bat wrappers call this file, so behaviour is identical on
every OS. Nothing here needs network access.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def venv_python() -> str:
    """Prefer the project's .venv interpreter; fall back to the current one."""

    candidates = (
        ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python"),
        ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def run(args: list[str]) -> int:
    print(f"[launch] {' '.join(args)}")
    return subprocess.call(args, cwd=str(ROOT))


def camera_settings(args: argparse.Namespace) -> tuple[str, str]:
    url = getattr(args, "camera_url", "") or os.environ.get("NORMEX_CAMERA_URL", "")
    snapshot = getattr(args, "snapshot_url", "") or os.environ.get("NORMEX_SNAPSHOT_URL", "")
    ip = getattr(args, "ip", "")
    if ip and not url and not snapshot:
        url = f"http://{ip}:{getattr(args, 'port', 8080)}/video"
    cfg_file = ROOT / "configs" / "camera.json"
    if not url and not snapshot and cfg_file.is_file():
        import json

        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            url = data.get("url", "")
            snapshot = data.get("snapshot_url", "")
        except (ValueError, OSError):
            pass
    return url, snapshot


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_dashboard(args: argparse.Namespace) -> int:
    url, snapshot = camera_settings(args)
    dashboard_port = getattr(args, "dashboard_port", 8765)
    cmd = [
        venv_python(), str(ROOT / "dashboard_server.py"),
        "--host", args.host, "--port", str(dashboard_port),
    ]
    if not getattr(args, "no_open", False):
        cmd.append("--open")
    if url:
        cmd += ["--camera-url", url]
    if snapshot:
        cmd += ["--snapshot-url", snapshot]
    if args.config:
        cmd += ["--config", args.config]
    return run(cmd)


def cmd_setup(args: argparse.Namespace) -> int:
    ok = True

    # 1. Synthetic demo (fast, no download)
    demo = ROOT / "datasets" / "demo"
    if not (demo / "good").is_dir():
        print("[setup] generating synthetic demo dataset…")
        code = run([venv_python(), str(ROOT / "scripts" / "make_demo_dataset.py"), "--out", str(demo)])
        ok = ok and code == 0
    else:
        print("[setup] synthetic demo dataset present")

    # 2. MVTec / converted datasets
    datasets = ROOT / "datasets"
    found = []
    if datasets.is_dir():
        for root in sorted(datasets.iterdir()):
            if not root.is_dir():
                continue
            for cat in sorted(p for p in root.iterdir() if p.is_dir()):
                if (cat / "train" / "good").is_dir():
                    found.append(f"{root.name}/{cat.name}")
    if found:
        print("[setup] trainable datasets: " + ", ".join(found))
    else:
        print("[setup] WARNING: no trainable datasets found (expected datasets/<root>/<category>/train/good)")

    # 3. Profiles
    profiles = ROOT / "profiles"
    names = sorted(p.name for p in profiles.iterdir() if p.is_dir()) if profiles.is_dir() else []
    print("[setup] profiles: " + (", ".join(names) if names else "none yet"))

    print("[setup] done. Start the dashboard with: python launch.py")
    return 0 if ok else 1


def cmd_train_demo(args: argparse.Namespace) -> int:
    good = ROOT / "datasets" / "demo" / "good"
    if not good.is_dir():
        print("[train-demo] demo dataset missing; run: python launch.py setup")
        return 1
    return run([venv_python(), str(ROOT / "main.py"), "train", "--product", "demo", "--images", str(good)])


def cmd_train_all(args: argparse.Namespace) -> int:
    cmd = [venv_python(), str(ROOT / "scripts" / "train_all.py"), "--config", args.config]
    if args.force:
        cmd.append("--force")
    if args.only:
        cmd += ["--only", args.only]
    return run(cmd)


def cmd_benchmark(args: argparse.Namespace) -> int:
    cmd = [
        venv_python(), str(ROOT / "main.py"), "validate",
        "--category", args.category,
        "--dataset", args.dataset,
        "--config", args.config,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return run(cmd)


def cmd_capture(args: argparse.Namespace) -> int:
    url, snapshot = camera_settings(args)
    cmd = [
        venv_python(), str(ROOT / "main.py"), "capture",
        "--out", args.out, "--count", str(args.count),
        "--interval", str(args.interval), "--prefix", args.prefix,
    ]
    if url:
        cmd += ["--url", url]
    if snapshot:
        cmd += ["--snapshot-url", snapshot]
    if args.ip:
        cmd += ["--ip", args.ip, "--port", str(args.port)]
    return run(cmd)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="launch.py", description="NORMEX launcher")
    sub = parser.add_subparsers(dest="command")

    camera_parent = argparse.ArgumentParser(add_help=False)
    camera_parent.add_argument("--camera-url", default="", help="IP camera stream URL")
    camera_parent.add_argument("--snapshot-url", default="", help="JPEG snapshot URL")
    camera_parent.add_argument("--ip", default="", help="phone LAN IP (builds http://IP:PORT/video)")
    camera_parent.add_argument("--port", type=int, default=8080, help="phone camera port")

    p_dash = sub.add_parser("dashboard", parents=[camera_parent], help="start the live dashboard (default)")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--dashboard-port", type=int, default=8765, help="dashboard HTTP port")
    p_dash.add_argument("--no-open", action="store_true", help="do not open the browser (service mode)")
    p_dash.add_argument("--config", default=None)
    p_dash.set_defaults(func=cmd_dashboard)

    p_setup = sub.add_parser("setup", help="verify datasets and generate the synthetic demo if missing")
    p_setup.set_defaults(func=cmd_setup)

    p_train = sub.add_parser("train-demo", help="train the synthetic demo profile")
    p_train.set_defaults(func=cmd_train_demo)

    p_all = sub.add_parser("train-all", help="train every dataset that has no profile yet")
    p_all.add_argument("--config", default="configs/mvtec_fast.yaml")
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--only", default="", help="comma-separated product ids")
    p_all.set_defaults(func=cmd_train_all)

    p_bench = sub.add_parser("benchmark", help="run the MVTec AD benchmark")
    p_bench.add_argument("--category", default="screw")
    p_bench.add_argument("--dataset", default="datasets/mvtec_ad")
    p_bench.add_argument("--config", default="configs/mvtec_fast.yaml")
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    p_cap = sub.add_parser("capture", parents=[camera_parent], help="capture good frames from a camera")
    p_cap.add_argument("--out", required=True)
    p_cap.add_argument("--count", type=int, default=30)
    p_cap.add_argument("--interval", type=float, default=0.5)
    p_cap.add_argument("--prefix", default="good")
    p_cap.set_defaults(func=cmd_capture)

    return parser


def _dashboard_with_port(args: argparse.Namespace) -> int:
    return cmd_dashboard(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default behaviour: setup check + dashboard
        ns = argparse.Namespace(
            command="dashboard", camera_url="", snapshot_url="", ip="", port=8080,
            host="127.0.0.1", dashboard_port=8765, config=None,
        )
        rc = cmd_setup(argparse.Namespace())
        return cmd_dashboard(ns) if rc == 0 else rc

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
