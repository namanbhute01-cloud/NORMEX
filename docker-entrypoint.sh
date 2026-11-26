#!/bin/sh
# NORMEX container entrypoint.
#
#   serve                 start the dashboard server on 0.0.0.0:$NORMEX_PORT
#   train <args...>       run main.py train ...
#   inspect <args...>     run main.py inspect ...
#   validate <args...>    run main.py validate ...
#   <anything else>       executed verbatim (e.g. bash)

set -e

case "$1" in
  serve)
    shift
    exec python dashboard_server.py \
      --host "${NORMEX_HOST:-0.0.0.0}" \
      --port "${NORMEX_PORT:-8765}" \
      ${NORMEX_CAMERA_URL:+--camera-url "$NORMEX_CAMERA_URL"} \
      ${NORMEX_SNAPSHOT_URL:+--snapshot-url "$NORMEX_SNAPSHOT_URL"} \
      "$@"
    ;;
  train|inspect|validate|list|rollback|capture)
    exec python main.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
