#!/usr/bin/env python3
"""Generate a deterministic synthetic demo dataset.

Writes ``good/``, ``new_good/``, ``defect/`` and ``masks/``.  Only ``good/``
is used for training; the other folders exist to inspect and score the flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_6_evaluation.synthetic import generate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/demo")
    parser.add_argument("--good", type=int, default=30, help="training-good images")
    parser.add_argument("--new-good", type=int, default=10, help="unseen good images")
    parser.add_argument("--defect", type=int, default=15, help="defective images")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if args.good < 5:
        parser.error("--good must be at least 5")
    if args.size < 64:
        parser.error("--size must be at least 64")

    info = generate_dataset(
        args.out,
        n_good=args.good,
        n_new_good=args.new_good,
        n_defect=args.defect,
        size=args.size,
        seed=args.seed,
    )
    print(f"[demo] generated {info['n_good']} good, {info['n_new_good']} unseen good, "
          f"{info['n_defect']} defect images at {info['size']}x{info['size']}")
    print(f"[demo] root : {info['root']}")
    print(f"[demo] train: {info['good']}")
    print(f"[demo] test : {info['new_good']} (good), {info['defect']} (defect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
