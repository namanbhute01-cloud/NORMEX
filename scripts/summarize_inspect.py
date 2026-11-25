#!/usr/bin/env python3
"""Summarize one or two ``inspect --jsonl`` files with real metrics.

Example:
    python scripts/summarize_inspect.py \
        --good results/demo/good.jsonl \
        --defect results/demo/defect.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_6_evaluation.metrics import far_frr, image_auroc, p95  # noqa: E402


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def _latency(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.array([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    return float(values.mean()), p95(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", required=True, help="JSONL from inspecting unseen good images")
    parser.add_argument("--defect", default=None, help="JSONL from inspecting defective images")
    parser.add_argument("--skip-first", action="store_true",
                        help="exclude the first row per file (cold-start latency)")
    args = parser.parse_args(argv)

    good_rows = _read(Path(args.good))
    defect_rows = _read(Path(args.defect)) if args.defect else []
    if args.skip_first:
        good_rows = good_rows[1:] or good_rows
        defect_rows = defect_rows[1:] or defect_rows

    all_rows = good_rows + defect_rows
    labels = [0] * len(good_rows) + [1] * len(defect_rows)
    scores = [float(row["score"]) for row in all_rows]
    statuses = [str(row["status"]) for row in all_rows]

    print(f"good images    : {len(good_rows)}")
    if defect_rows:
        print(f"defect images  : {len(defect_rows)}")
    print(f"image AUROC    : {image_auroc(labels, scores):.4f}" if defect_rows else "image AUROC    : n/a (no defect file)")

    good_scores = np.array([float(row["score"]) for row in good_rows])
    print(f"good score     : mean {good_scores.mean():.3f}  range {good_scores.min():.3f}..{good_scores.max():.3f}")
    if defect_rows:
        defect_scores = np.array([float(row["score"]) for row in defect_rows])
        print(f"defect score   : mean {defect_scores.mean():.3f}  range {defect_scores.min():.3f}..{defect_scores.max():.3f}")
        rates = far_frr(statuses, labels)
        flagged = sum(status in {"DEFECTIVE", "REVIEW"} for status in statuses[len(good_rows):])
        print(f"defects flagged: {flagged}/{len(defect_rows)} ({flagged / len(defect_rows):.0%})")
        print(f"good -> DEFECTIVE: {sum(row['status'] == 'DEFECTIVE' for row in good_rows)}/{len(good_rows)}")
        print(f"FAR / FRR      : {rates['far']:.3f} / {rates['frr']:.3f}")

    for name, rows in (("good", good_rows), ("defect", defect_rows)):
        if not rows:
            continue
        lat_mean, lat_p95 = _latency(rows, "latency_ms")
        pre_mean, pre_p95 = _latency(rows, "preprocess_ms")
        print(f"{name:<6} latency: {lat_mean:.1f} ms mean / {lat_p95:.1f} ms P95"
              f"   preprocess: {pre_mean:.1f} / {pre_p95:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
