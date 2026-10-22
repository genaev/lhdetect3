#!/usr/bin/env python3
"""Count trichomes and measure their lengths with RF-DETR-Seg.

The checkpoint is the soybean + wheat RF-DETR-Seg Small model. For every
image the script writes one CSV row with the instance count and the
centerline length of each trichome, in pixels.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from centerline import _Instance, _compute_centerline, _fill_holes, _merge_instances

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO_DIR / "weights" / "rfdetrseg_small_wheat_soy.pth"
DEFAULT_CONFIDENCE = 0.35
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict trichome counts and centerline lengths."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO_DIR / "examples",
        help="Image file or directory. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_DIR / "predictions.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="RF-DETR-Seg Small checkpoint.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold. Default: {DEFAULT_CONFIDENCE}.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index; -1 runs on CPU.")
    parser.add_argument("--min-area", type=float, default=20.0, help="Minimal mask area, px.")
    parser.add_argument("--contour-dilate-px", type=int, default=3, help="Isotropic mask dilation before skeletonization, px.")
    parser.add_argument("--dilate-px", type=int, default=5, help="Dilation along the principal axis of a mask, px.")
    parser.add_argument("--smooth-sigma", type=float, default=6.0, help="Gaussian smoothing of the centerline, px.")
    parser.add_argument("--trim-angle", type=float, default=60.0, help="Trim sharp bends at centerline ends, degrees.")
    parser.add_argument("--merge-dist-px", type=float, default=60.0, help="Maximal gap between merged fragments, px.")
    parser.add_argument("--merge-max-angle", type=float, default=30.0, help="Maximal angle between merged fragments, degrees.")
    parser.add_argument("--merge-max-turn", type=float, default=40.0, help="Maximal direction change at a merge junction, degrees.")
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    found: list[Path] = []
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                found.append(Path(dirpath) / name)
    return sorted(found)


def load_model(checkpoint: Path, device: str):
    import torch
    import rfdetr

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "args" not in state or "model" not in state:
        raise RuntimeError(f"Checkpoint has no args/model payload: {checkpoint}")

    saved = state["args"]
    num_classes = int(state["model"]["class_embed.bias"].shape[0]) - 1
    model = rfdetr.RFDETRSegSmall(
        resolution=int(getattr(saved, "resolution", 1272)),
        num_classes=num_classes,
        positional_encoding_size=int(getattr(saved, "positional_encoding_size", 32)),
        segmentation_head=True,
        pretrain_weights=str(checkpoint),
        device=device,
    )
    model.model.model.eval()
    return model


def load_rgb(image_path: Path):
    import cv2
    import numpy as np

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    buffer = np.fromfile(image_path, dtype=np.uint8)
    if buffer.size == 0 or not buffer.any():
        return None
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def instance_lengths(mask_stack, args) -> list[float]:
    import cv2
    import numpy as np

    if mask_stack is None or len(mask_stack) == 0:
        return []

    instances = []
    for mask in mask_stack:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask_u8 = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        mask_u8 = _fill_holes(mask_u8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < args.min_area:
            continue
        length, path = _compute_centerline(
            mask_u8,
            contour,
            args.contour_dilate_px,
            args.dilate_px,
            smooth_sigma=args.smooth_sigma,
            trim_angle_deg=args.trim_angle,
        )
        instances.append(_Instance(mask=mask_u8, contour=contour, path=path, length=length))

    instances = _merge_instances(
        instances,
        args.merge_dist_px,
        args.merge_max_angle,
        max_turn=args.merge_max_turn,
        smooth_sigma=args.smooth_sigma,
        trim_angle_deg=args.trim_angle,
    )
    return [float(instance.length) for instance in instances if not instance.merged]


def predict_lengths(model, image_path: Path, args) -> list[float] | None:
    image = load_rgb(image_path)
    if image is None:
        return None
    detections = model.predict(image, threshold=args.conf)
    if detections is None or len(detections) == 0:
        return []
    return instance_lengths(detections.mask, args)


def completed_images(output: Path) -> set[str]:
    if not output.is_file() or output.stat().st_size == 0:
        return set()
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["image"] for row in reader if row.get("image")}


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if args.gpu >= 0 else "cpu"

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    images = iter_images(input_path)
    if not images:
        raise SystemExit(f"No images found: {input_path}")

    root = input_path if input_path.is_dir() else input_path.parent
    done = completed_images(output)
    pending = [
        path for path in images
        if path.relative_to(root).as_posix() not in done
    ]
    print(f"Images: {len(images)}; already written: {len(done)}; confidence: {args.conf}")
    print(f"Weights: {args.weights}")
    if not pending:
        print(f"CSV is already complete: {output}")
        return

    model = load_model(args.weights, device)
    write_header = not done
    skipped = 0
    with output.open("a" if done else "w", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["image", "count", "lengths_px"])
            handle.flush()
        for index, image_path in enumerate(images, start=1):
            relative = image_path.relative_to(root).as_posix()
            if relative in done:
                continue
            lengths = predict_lengths(model, image_path, args)
            if lengths is None:
                skipped += 1
                print(f"[{index}/{len(images)}] skip unreadable file: {relative}")
                continue
            formatted = ";".join(f"{length:.1f}" for length in lengths)
            writer.writerow([relative, len(lengths), formatted])
            handle.flush()
            print(f"[{index}/{len(images)}] {relative}: count={len(lengths)}")

    print(f"CSV: {output}")
    if skipped:
        print(f"Unreadable files skipped: {skipped}")


if __name__ == "__main__":
    sys.exit(main())
