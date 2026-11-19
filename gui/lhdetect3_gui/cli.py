"""Command-line version: lhdetect3-cli <folder> [--scale UM_PER_PX] [--output results.csv]."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .engine import DEFAULT_CONFIDENCE, Engine, Params, list_images
from .batch import done_images, run_batch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count trichomes and measure their lengths (LHDetect3).")
    ap.add_argument("input", type=Path, help="image file or folder (scanned recursively)")
    ap.add_argument("--output", type=Path, help="CSV file (default: <folder>/lhdetect3_results.csv)")
    ap.add_argument("--scale", type=float, default=0.0, help="micrometers per pixel; 0 = report pixels")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="confidence threshold")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--annotated", type=Path, help="folder for images with drawn contours")
    ap.add_argument("--model", type=Path, help="ONNX model (default: bundled)")
    ap.add_argument("--resume", action="store_true", help="skip images already in the CSV")
    a = ap.parse_args(argv)

    root = a.input if a.input.is_dir() else a.input.parent
    out = a.output or (root / "lhdetect3_results.csv")
    files = list_images(a.input)
    if not files:
        print(f"No images found: {a.input}", file=sys.stderr)
        return 1
    if a.resume:
        done = done_images(out)
        files = [f for f in files if f.relative_to(root).as_posix() not in done]
    engine = Engine(a.model, device=a.device)
    t0 = time.time()

    def progress(i, n, res):
        el = time.time() - t0
        eta = el / i * (n - i)
        note = res.error or f"count={res.count}"
        print(f"[{i}/{n}] {res.image}: {note}  ({res.seconds:.1f} s; about {eta / 60:.1f} min left)", flush=True)

    run_batch(engine, files, root, Params(confidence=a.conf), a.scale, out, append=a.resume,
              annotated_dir=a.annotated, on_result=progress)
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
