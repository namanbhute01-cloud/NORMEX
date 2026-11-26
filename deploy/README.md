# Deploying NORMEX

NORMEX is a single-process, offline-first CPU application:

```
camera (phone / USB / RTSP)
        │
        ▼
dashboard_server.py ──► dashboard.html   (animated welcome → dashboard)
        │
        ├── ProfileStore (profiles/<product>/vNNN)
        └── results/live/*.jsonl + PNG heatmaps
```

The server binds `127.0.0.1` by default. It has **no authentication**, so do not
bind it to `0.0.0.0` on an untrusted network. If operators need remote access,
put it behind a reverse proxy with TLS and auth (nginx/Caddy) or an SSH tunnel.

---

## Linux — systemd service

```bash
sudo useradd --system --home /opt/normex --shell /usr/sbin/nologin normex
sudo mkdir -p /opt/normex
sudo rsync -a --exclude .venv --exclude datasets --exclude profiles ./ /opt/normex/main_folder/
# Copy datasets/profiles separately if they are large:
sudo rsync -a datasets profiles /opt/normex/main_folder/
sudo chown -R normex:normex /opt/normex

# create the venv as the service user (see README_QUICKSTART.md), then:
sudo cp main_folder/deploy/normex.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now normex
systemctl status normex
curl http://127.0.0.1:8765/healthz
```

Logs: `journalctl -u normex -f`.  The service restarts on crash.

To train while the service is running, either stop it first (training needs the
CPU and up to ~4 GB RAM) or accept slower inspection during training.

---

## Windows — Task Scheduler

From an elevated PowerShell in the project root:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
```

The dashboard starts at logon on <http://127.0.0.1:8765> and restarts up to
three times on failure. Use `-AtStartup` for a kiosk machine with auto-logon.

Remove it with:

```powershell
Unregister-ScheduledTask -TaskName "NORMEX dashboard" -Confirm:$false
```

For a single interactive session, `start.bat` is still the simplest option: it
opens the dashboard in the default browser.

---

## Docker / Compose

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8765/healthz
```

`docker-compose.yml` mounts `datasets/`, `profiles/`, `results/` and `configs/`
so training and camera settings survive container upgrades. The healthcheck
polls `/healthz` every 30 s and the container restarts unless-stopped.

Camera URL precedence inside the container:

1. `NORMEX_CAMERA_URL` / `NORMEX_SNAPSHOT_URL`
2. `configs/camera.json` (written by the dashboard's Live Camera tab; loopback only)
3. no camera (the Results/Datasets tabs still work)

Note: `configs/camera.json` is written inside the mounted `configs/` volume, so
connecting from the Live Camera tab works the same in Docker.

---

## Health and monitoring

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness probe, returns `{"ok": true}` in milliseconds |
| `GET /api/status` | datasets, profiles, latest benchmark, CPU/RAM |
| `GET /api/results` | recent verdicts from `results/**/*.jsonl` |

Every live inspection appends one JSON line to `results/live/live.jsonl` and a
PNG heatmap next to it. Ship `results/` to your MES/SCADA system, or scrape
`/api/results` — no database is required.

---

## Production checklist (read before claiming readiness)

- [x] Versioned, atomic profiles; camera URL persisted across restarts
- [x] Service supervision (systemd / Task Scheduler / Docker restart policy)
- [x] Health endpoint and structured per-inspection JSONL
- [x] Works offline on Windows and Linux; Docker image supplied
- [ ] **Accuracy acceptance on your own parts** — train, then run a held-out
      set with known defects; require AUROC ≥ 0.95, missed defects ≤ 10%
- [ ] **Latency acceptance on your target laptop** — `scripts/bench_latency.py`
      must report P95 < 1000 ms with the production config
- [ ] Camera mount and lighting frozen; capture ≥ 30 good images from the same
      viewpoint
- [ ] Network isolation / reverse-proxy auth if the dashboard leaves localhost
