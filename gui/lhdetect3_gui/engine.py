"""Trichome detection with the RF-DETR-Seg Small model exported to ONNX.

The pre- and post-processing reproduce ``RFDETR.predict`` from rfdetr 1.5.2:
ImageNet normalization, antialiased bilinear resize to the model resolution,
sigmoid scores, top-k selection, bilinear upsampling of the selected mask
logits to the image size and binarization at logit > 0.

The execution device is chosen automatically: the first GPU execution
provider that loads the model and reproduces the CPU result on a reference
image is used; otherwise the CPU is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import centerline as cl

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CONFIDENCE = 0.37
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# GPU execution providers in order of preference.
GPU_PROVIDERS = (
    "CUDAExecutionProvider",       # NVIDIA (Linux, Windows)
    "ROCMExecutionProvider",       # AMD (Linux)
    "MIGraphXExecutionProvider",   # AMD (Linux)
    "DmlExecutionProvider",        # DirectX 12: AMD, NVIDIA, Intel (Windows)
    "CoreMLExecutionProvider",     # Apple (macOS)
)
PROVIDER_LABELS = {
    "CUDAExecutionProvider": "GPU (NVIDIA CUDA)",
    "ROCMExecutionProvider": "GPU (AMD ROCm)",
    "MIGraphXExecutionProvider": "GPU (AMD MIGraphX)",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CoreMLExecutionProvider": "GPU (Apple Core ML)",
    "CPUExecutionProvider": "CPU",
}


@dataclass
class Params:
    """Post-processing parameters (the same defaults as in infer.py)."""
    confidence: float = DEFAULT_CONFIDENCE
    min_area: float = 20.0                   # smaller masks are counted but not measured
    smooth_sigma: float = cl.SIGMA


@dataclass
class Trichome:
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2, image pixels
    score: float
    contour: Optional[np.ndarray] = None     # (N, 2) int32; None if the mask is empty
    path: List[Tuple[float, float]] = field(default_factory=list)   # centerline
    length_px: Optional[float] = None        # None if the mask is too small to measure


@dataclass
class ImageResult:
    image: str                               # path relative to the input folder
    width: int
    height: int
    trichomes: List[Trichome] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def count(self) -> int:
        """Number of detected trichomes (the quantity validated in the paper)."""
        return len(self.trichomes)

    @property
    def n_measured(self) -> int:
        return sum(t.length_px is not None for t in self.trichomes)

    def lengths(self, um_per_px: float = 0.0) -> List[float]:
        """Lengths of the trichomes whose mask could be measured."""
        k = um_per_px if um_per_px > 0 else 1.0
        return [t.length_px * k for t in self.trichomes if t.length_px is not None]


# --------------------------------------------------------------------------- files

def resource_dir() -> Path:
    """Folder with bundled data (PyInstaller-aware)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "lhdetect3_gui"
    return Path(__file__).resolve().parent


def default_model_path() -> Path:
    env = os.environ.get("LHDETECT3_MODEL")
    if env:
        return Path(env)
    return resource_dir() / "models" / "lhdetect3_rfdetr_seg_small.onnx"


def list_images(folder: Path, recursive: bool = True) -> List[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    found: List[Path] = []
    if recursive:
        for dirpath, _, names in os.walk(folder):
            for name in names:
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    found.append(Path(dirpath) / name)
    else:
        found = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(found)


def load_rgb(path: Path) -> Optional[np.ndarray]:
    """RGB uint8 array or None. Works with non-ASCII paths."""
    import cv2
    try:
        with Image.open(path) as im:
            if im.mode in ("I;16", "I;16B", "I", "F"):
                arr = np.asarray(im, dtype=np.float32)
                arr = (255.0 * (arr - arr.min()) / max(1e-6, float(arr.max() - arr.min()))).astype(np.uint8)
                return np.dstack([arr] * 3)
            return np.asarray(im.convert("RGB"))
    except Exception:
        pass
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- engine

def _cache_file() -> Path:
    if sys.platform.startswith("win"):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "LHDetect3" / "device_check.json"


class Engine:
    """ONNX Runtime session with automatic GPU/CPU selection."""

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto",
                 log: Callable[[str], None] = print):
        import onnxruntime as ort

        self.ort = ort
        self.log = log
        self.model_path = Path(model_path or default_model_path())
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.session = None
        self.provider = "CPUExecutionProvider"
        self._select(device)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.resolution = int(inp.shape[-1])
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ---- device selection
    @property
    def device_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    def _make_session(self, provider: str):
        ort = self.ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        providers: list = [provider]
        if provider == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if provider == "CoreMLExecutionProvider":
            providers = [(provider, {"MLComputeUnits": "CPUAndGPU"})]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def _reference_input(self) -> np.ndarray:
        ref = resource_dir() / "models" / "reference.jpg"
        rgb = load_rgb(ref) if ref.is_file() else None
        if rgb is None:
            rng = np.random.default_rng(0)
            rgb = (rng.random((self.resolution, self.resolution, 3)) * 255).astype(np.uint8)
        return self._preprocess(rgb)

    def _model_key(self) -> str:
        st = self.model_path.stat()
        return f"{self.model_path.name}:{st.st_size}:{int(st.st_mtime)}:{self.ort.__version__}"

    def _select(self, device: str) -> None:
        available = list(self.ort.get_available_providers())
        candidates = [] if device == "cpu" else [p for p in GPU_PROVIDERS if p in available]
        self.log(f"ONNX Runtime {self.ort.__version__}; providers: {', '.join(available)}")
        cpu = self._make_session("CPUExecutionProvider")
        self.session, self.provider = cpu, "CPUExecutionProvider"
        if not candidates:
            self.log("Inference device: CPU")
            return
        inp = cpu.get_inputs()[0]
        self.input_name = inp.name
        self.resolution = int(inp.shape[-1])

        cache_path = _cache_file()
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
        key = self._model_key()
        x = None
        ref_out = None
        for prov in candidates:
            ck = f"{key}:{prov}"
            if cache.get(ck) is False:
                self.log(f"{PROVIDER_LABELS[prov]}: skipped (failed the check earlier)")
                continue
            try:
                sess = self._make_session(prov)
                if prov not in sess.get_providers():
                    raise RuntimeError("provider was not enabled")
                if cache.get(ck) is not True:
                    if x is None:
                        x = self._reference_input()
                        ref_out = cpu.run(None, {self.input_name: x})
                    out = sess.run(None, {self.input_name: x})
                    if not self._outputs_agree(ref_out, out):
                        cache[ck] = False
                        raise RuntimeError("results differ from the CPU")
                    cache[ck] = True
                self.session, self.provider = sess, prov
                self.log(f"Inference device: {PROVIDER_LABELS[prov]}")
                break
            except Exception as exc:  # noqa: BLE001
                self.log(f"{PROVIDER_LABELS[prov]} is not usable ({exc}); trying the next device")
        else:
            self.log("Inference device: CPU")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
        except Exception:
            pass

    @staticmethod
    def _outputs_agree(a, b, thr: float = 0.2) -> bool:
        """Same detections above a low score threshold and nearly equal masks."""
        sa = 1 / (1 + np.exp(-a[1][0].max(-1)))
        sb = 1 / (1 + np.exp(-b[1][0].max(-1)))
        if np.abs(sa - sb).max() > 0.02:
            return False
        keep = (sa > thr) | (sb > thr)
        if keep.sum() == 0:
            return True
        ma = a[2][0][keep] > 0
        mb = b[2][0][keep] > 0
        inter = np.logical_and(ma, mb).sum()
        union = np.logical_or(ma, mb).sum()
        return union == 0 or inter / union > 0.97

    def fallback_to_cpu(self) -> None:
        self.session = self._make_session("CPUExecutionProvider")
        self.provider = "CPUExecutionProvider"
        self.log("Switched to CPU")

    # ---- inference
    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        r = self.resolution
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        chans = [np.asarray(Image.fromarray(np.ascontiguousarray(x[..., c]), mode="F")
                            .resize((r, r), Image.BILINEAR), dtype=np.float32) for c in range(3)]
        return np.stack(chans, 0)[None]

    def detect(self, rgb: np.ndarray, confidence: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scores (K,), boxes (K, 4; x1, y1, x2, y2) and binary masks (K, H, W) uint8 of the kept detections."""
        import cv2
        h, w = rgb.shape[:2]
        x = self._preprocess(rgb)
        try:
            dets, logits, masks = self.session.run(None, {self.input_name: x})
        except Exception:
            if self.provider == "CPUExecutionProvider":
                raise
            self.log(f"{self.device_label} failed during inference; switching to CPU")
            self.fallback_to_cpu()
            dets, logits, masks = self.session.run(None, {self.input_name: x})
        logits = logits[0]
        masks = masks[0]
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        flat = prob.reshape(-1)
        k = min(100, flat.size)
        idx = np.argsort(-flat, kind="stable")[:k]
        scores = flat[idx]
        queries = idx // logits.shape[1]
        keep = scores > confidence
        scores, queries = scores[keep], queries[keep]
        cx, cy, bw, bh = dets[0][queries].T if len(queries) else np.zeros((4, 0))
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        out = np.zeros((len(queries), h, w), dtype=np.uint8)
        for i, q in enumerate(queries):
            up = cv2.resize(masks[q].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            out[i] = (up > 0.0).astype(np.uint8) * 255
        return scores.astype(np.float32), boxes.reshape(-1, 4), out

    def process(self, rgb: np.ndarray, p: Params) -> List[Trichome]:
        """Every detection is a trichome; its length is measured from the mask centerline."""
        import cv2
        scores, boxes, masks = self.detect(rgb, p.confidence)
        result = []
        for score, box, m in zip(scores, boxes, masks):
            t = Trichome(box=tuple(float(v) for v in box), score=float(score))
            m = cl.fill_holes(m)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                t.contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
                if cv2.countNonZero(m) >= p.min_area:
                    length, path = cl.measure_mask(m, sigma=p.smooth_sigma)
                    if length > 0:
                        t.length_px, t.path = float(length), [(float(x), float(y)) for x, y in path]
            result.append(t)
        return result

    def run_file(self, path: Path, root: Path, p: Params) -> ImageResult:
        t0 = time.perf_counter()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        rgb = load_rgb(path)
        if rgb is None:
            return ImageResult(rel, 0, 0, error="unreadable image")
        h, w = rgb.shape[:2]
        try:
            tr = self.process(rgb, p)
        except Exception as exc:  # noqa: BLE001
            return ImageResult(rel, w, h, error=str(exc), seconds=time.perf_counter() - t0)
        return ImageResult(rel, w, h, trichomes=tr, seconds=time.perf_counter() - t0)


def system_info() -> str:
    return f"{platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}"
