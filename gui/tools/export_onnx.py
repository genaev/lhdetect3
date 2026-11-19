#!/usr/bin/env python3
"""Export the RF-DETR-Seg checkpoint to ONNX for the LHDetect3 application.

Run in an environment with torch and rfdetr==1.5.2 (the one used for training),
with about 16 GB of RAM:

    pip install onnx onnxruntime          # onnxruntime only for the check
    python tools/export_onnx.py ../weights/rfdetrseg_small_wheat_soy.pth \
        lhdetect3_gui/models/lhdetect3_rfdetr_seg_small.onnx --check ../examples

Self-attention over the whole image (four global layers of the DINOv2
backbone) is exported in 16 query chunks. The result is mathematically the
same, but the 6 x 11,236 x 11,236 attention matrix is never stored at once,
so inference needs about 1-2 GB of memory instead of 6 GB.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_CHUNKS = 16


def sdpa_chunked(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if attn_mask is not None or is_causal:
        raise NotImplementedError
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    kt = k.transpose(-2, -1)
    n = q.shape[-2]
    if n <= 4096:
        return torch.softmax(torch.matmul(q, kt) * s, dim=-1).matmul(v)
    size = int(math.ceil(n / N_CHUNKS))
    parts = [torch.softmax(torch.matmul(qc, kt) * s, dim=-1).matmul(v) for qc in torch.split(q, size, dim=-2)]
    return torch.cat(parts, dim=-2)


def load(checkpoint: Path):
    import rfdetr
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = state["args"]
    num_classes = int(state["model"]["class_embed.bias"].shape[0]) - 1
    model = rfdetr.RFDETRSegSmall(
        resolution=int(getattr(saved, "resolution", 1272)),
        num_classes=num_classes,
        positional_encoding_size=int(getattr(saved, "positional_encoding_size", 32)),
        segmentation_head=True,
        pretrain_weights=str(checkpoint),
        device="cpu",
    )
    model.model.model.eval()
    return model


def torch_reference(model, image_path: Path, conf: float):
    """Scores and binary masks exactly as RFDETR.predict, upsampling only kept masks."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    res = model.model.resolution
    img = TF.to_tensor(Image.open(image_path).convert("RGB"))
    h, w = img.shape[1:]
    x = TF.resize(TF.normalize(img, model.means, model.stds), (res, res)).unsqueeze(0)
    with torch.no_grad():
        out = model.model.model(x)
    prob = out["pred_logits"].sigmoid()[0]
    val, idx = torch.topk(prob.reshape(-1), min(100, prob.numel()))
    q = idx // prob.shape[1]
    keep = val > conf
    masks = F.interpolate(out["pred_masks"][0][q[keep]].unsqueeze(1), size=(h, w),
                          mode="bilinear", align_corners=False)[:, 0] > 0
    return val[keep].numpy(), masks.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--check", type=Path, help="folder with test images: compare ONNX Runtime with PyTorch")
    ap.add_argument("--conf", type=float, default=0.37)
    ap.add_argument("--resolution", type=int, help=argparse.SUPPRESS)  # testing only
    args = ap.parse_args()

    model = load(args.checkpoint)
    net = model.model.model
    if args.resolution:
        model.model.resolution = args.resolution
        for mod in net.modules():
            if hasattr(mod, "export") and isinstance(getattr(mod, "shape", None), (tuple, list)):
                mod.shape = (args.resolution, args.resolution)
    res = model.model.resolution
    print(f"Resolution {res}; exporting...", flush=True)

    import copy
    export_net = copy.deepcopy(net).eval()
    export_net.export()
    original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa_chunked
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        torch.onnx.export(
            export_net, torch.randn(1, 3, res, res), str(args.output),
            input_names=["input"], output_names=["dets", "labels", "masks"],
            export_params=True, keep_initializers_as_inputs=False, do_constant_folding=True,
            opset_version=17, dynamic_axes=None, dynamo=False,
        )
    finally:
        F.scaled_dot_product_attention = original_sdpa
    del export_net
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s", flush=True)

    if not args.check:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lhdetect3_gui.engine import Engine, IMAGE_EXTENSIONS, load_rgb
    engine = Engine(args.output, device="cpu", log=lambda s: None)
    images = sorted(p for p in args.check.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    worst = 1.0
    same_counts = True
    for path in images:
        ref_scores, ref_masks = torch_reference(model, path, args.conf)
        scores, masks = engine.detect(load_rgb(path), args.conf)
        ious = []
        for a in masks > 0:
            if a.sum() == 0 and any(b.sum() == 0 for b in ref_masks):
                ious.append(1.0)
                continue
            ious.append(max(((a & b).sum() / max(1, (a | b).sum())) for b in ref_masks) if len(ref_masks) else 0.0)
        miou = min(ious) if ious else 1.0
        worst = min(worst, miou)
        same_counts &= len(scores) == len(ref_scores)
        print(f"{path.name}: detections ONNX {len(scores)}, PyTorch {len(ref_scores)}; "
              f"min mask IoU {miou:.4f}", flush=True)
    print("CHECK PASSED" if worst > 0.98 and same_counts else "CHECK FAILED")


if __name__ == "__main__":
    main()
