#!/usr/bin/env bash
# start.sh — NORMEX launcher (Linux / macOS)
#
#   ./start.sh                                   start the live dashboard
#   ./start.sh dashboard --camera-url http://192.168.1.100:8080/video
#   ./start.sh setup                             verify datasets / create demo
#   ./start.sh benchmark                         run the MVTec AD benchmark
#   ./start.sh capture --out datasets/mygood --camera-url http://IP:8080/video
#
# Everything is delegated to launch.py so Windows and Linux behave the same.

set -euo pipefail
cd "$(dirname "$0")"

# Pick the project interpreter.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "error: no Python interpreter found (create .venv or install python3)" >&2
    exit 1
fi

exec "$PY" launch.py "$@"
