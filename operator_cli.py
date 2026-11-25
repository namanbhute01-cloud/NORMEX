#!/usr/bin/env python3
"""NORMEX — Complete Operator Interface

A comprehensive CLI that lets an operator:
  1. Select a product/mode (MVTec AD or Synthetic)
  2. Learn from good images (train a profile)
  3. Inspect a new image (defect detection)
  4. View summary dashboard
  5. Run the full benchmark

This is the main entry point for the NORMEX operator.
"""

import argparse
import sys
from pathlib import Path


def cmd_operator_train(args):
    """Train a product profile from good images."""
    product_id = args.product
    images_dir = Path(args.images)
    config_path = args.config
    
    if not images_dir.is_dir():
        print(f"❌ Good images directory not found: {images_dir}")
        return 2
    
    # Use main.py train command
    from subprocess import run
    result = run([
        sys.executable, "main.py", "train",
        "--product", product_id,
        "--images", str(images_dir),
    ] + (["--config", config_path] if config_path else []))
    
    return result.returncode


def cmd_operator_inspect(args):
    """Inspect an image with a trained product."""
    product_id = args.product
    image_path = Path(args.image)
    config_path = args.config
    
    if not image_path.exists():
        print(f"❌ Inspection image not found: {image_path}")
        return 2
    
    from subprocess import run
    result = run([
        sys.executable, "main.py", "inspect",
        "--product", product_id,
        "--image", str(image_path),
    ] + (["--config", config_path] if config_path else []))
    
    return result.returncode


def cmd_operator_summary(args):
    """Show inspection summary."""
    good_file = args.good
    defect_file = args.defect
    
    from subprocess import run
    result = run([
        sys.executable, "scripts/summarize_inspect.py",
        "--good", good_file,
        "--defect", defect_file,
    ])
    
    return result.returncode


def cmd_operator_mvtec(args):
    """Run MVTec AD benchmark."""
    category = args.category
    
    from subprocess import run
    result = run([
        sys.executable, "main.py", "validate",
        "--category", category,
    ])
    
    return result.returncode


def cmd_operator_synthetic(args):
    """Run synthetic demo flow."""
    from subprocess import run
    
    # Step 1: Generate synthetic dataset
    result = run([sys.executable, "scripts/make_demo_dataset.py", "--out", "datasets/demo"])
    if result.returncode != 0:
        print("❌ Failed to generate synthetic dataset")
        return result.returncode
    
    # Step 2: Train
    result = run([sys.executable, "main.py", "train", "--product", args.product, "--images", "datasets/demo/good"])
    if result.returncode != 0:
        print("❌ Training failed")
        return result.returncode
    
    # Step 3: Inspect defects
    result = run([sys.executable, "main.py", "inspect", "--product", args.product, 
                  "--image", "datasets/demo/defect", "--out", "results/demo/defect",
                  "--jsonl", "results/demo/defect.jsonl"])
    if result.returncode != 0:
        print("❌ Inspection failed")
        return result.returncode
    
    # Step 4: Summary
    result = run([sys.executable, "scripts/summarize_inspect.py", 
                  "--good", "results/demo/good.jsonl",
                  "--defect", "results/demo/defect.jsonl"])
    
    return result.returncode


def cmd_operator_full(args):
    """Full operator flow: check dataset → train → inspect → summary."""
    product_id = args.product
    good_dir = Path(args.good_images)
    
    # Check if MVTec AD is available
    mvtec_root = Path("datasets/mvtec_ad")
    has_mvtec = mvtec_root.is_dir() and any(mvtec_root.iterdir())
    
    print(f"\n{'='*60}")
    print(f"  NORMEX OPERATOR INTERFACE")
    print(f"{'='*60}\n")
    
    if has_mvtec:
        print(f"✅ MVTec AD dataset found!")
        print(f"   Running benchmark for: {args.category}\n")
        # Run MVTec validate
        from subprocess import run
        result = run([sys.executable, "main.py", "validate", "--category", args.category])
        return result.returncode
    else:
        print(f"⚠️  MVTec AD dataset not found")
        print(f"   Using synthetic demo dataset\n")
    
    # Phase 1: Train
    print(f"\n[1/3] TRAINING")
    print(f"   Product: {product_id}")
    print(f"   Images: {good_dir}")
    
    if not good_dir.is_dir():
        print(f"❌ Good images directory not found: {good_dir}")
        return 2
    
    from subprocess import run
    result = run([sys.executable, "main.py", "train", "--product", product_id, "--images", str(good_dir)])
    if result.returncode != 0:
        print("❌ Training failed")
        return result.returncode
    
    # Phase 2: Inspect
    print(f"\n[2/3] INSPECTION")
    inspect_path = Path(args.image) if args.image else None
    
    if inspect_path and inspect_path.exists():
        print(f"   Image: {inspect_path}")
        result = run([sys.executable, "main.py", "inspect", "--product", product_id, "--image", str(inspect_path)])
        if result.returncode != 0:
            print("❌ Inspection failed")
            return result.returncode
    else:
        print(f"   (Skipping inspection - no image specified)")
        print(f"   Use --image <path> to inspect a specific image")
    
    # Phase 3: Summary
    print(f"\n[3/3] SUMMARY")
    good_jsonl = Path("results") / product_id / "good.jsonl"
    defect_jsonl = Path("results") / product_id / "defect.jsonl"
    
    if good_jsonl.exists() and defect_jsonl.exists():
        result = run([sys.executable, "scripts/summarize_inspect.py", 
                      "--good", str(good_jsonl), "--defect", str(defect_jsonl)])
    else:
        print("   (No summary data available)")
        print("   Run inspection first to generate results")
    
    print(f"\n{'='*60}")
    print(f"  NORMEX — Session Complete")
    print(f"{'='*60}\n")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="normex_operator.py",
        description="NORMEX Phase 1 Operator Interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    p_train = sub.add_parser("train", help="Train a product profile from good images")
    p_train.add_argument("--product", required=True, help="Product ID")
    p_train.add_argument("--images", required=True, help="Folder of good images")
    p_train.add_argument("--config", default=None, help="Config YAML path")
    p_train.set_defaults(func=cmd_operator_train)
    
    # Inspect command
    p_inspect = sub.add_parser("inspect", help="Inspect an image with a trained product")
    p_inspect.add_argument("--product", required=True, help="Product ID")
    p_inspect.add_argument("--image", required=True, help="Image file or folder to inspect")
    p_inspect.add_argument("--config", default=None, help="Config YAML path")
    p_inspect.set_defaults(func=cmd_operator_inspect)
    
    # Summary command
    p_summary = sub.add_parser("summary", help="Show inspection summary")
    p_summary.add_argument("--good", required=True, help="Good JSONL results file")
    p_summary.add_argument("--defect", required=True, help="Defect JSONL results file")
    p_summary.set_defaults(func=cmd_operator_summary)
    
    # MVTec benchmark command
    p_mvtec = sub.add_parser("mvtec", help="Run MVTec AD benchmark")
    p_mvtec.add_argument("--category", required=True, help="MVTec category (e.g., screw, bottle)")
    p_mvtec.set_defaults(func=cmd_operator_mvtec)
    
    # Synthetic demo command
    p_synthetic = sub.add_parser("synthetic", help="Run synthetic demo flow")
    p_synthetic.add_argument("--product", required=True, help="Product name for synthetic demo")
    p_synthetic.set_defaults(func=cmd_operator_synthetic)
    
    # Full operator flow command
    p_full = sub.add_parser("full", help="Full operator flow: check dataset → train → inspect → summary")
    p_full.add_argument("--product", required=True, help="Product ID")
    p_full.add_argument("--good-images", required=True, help="Folder of good training images")
    p_full.add_argument("--image", default=None, help="Image file to inspect")
    p_full.add_argument("--category", default="screw", help="MVTec category (default: screw)")
    p_full.set_defaults(func=cmd_operator_full)
    
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())