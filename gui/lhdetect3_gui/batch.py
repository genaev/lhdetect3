"""Batch processing of an image folder: CSV output and annotated images."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

from .engine import Engine, ImageResult, Params
from . import viz


def unit_name(um_per_px: float) -> str:
    return "um" if um_per_px > 0 else "px"


def csv_header(um_per_px: float) -> List[str]:
    u = unit_name(um_per_px)
    return ["image", "count", "n_measured", f"mean_length_{u}", f"lengths_{u}"]


def csv_row(res: ImageResult, um_per_px: float) -> List[str]:
    lengths = res.lengths(um_per_px)
    mean = f"{sum(lengths) / len(lengths):.2f}" if lengths else ""
    return [res.image, str(res.count), str(res.n_measured), mean, ";".join(f"{v:.2f}" for v in lengths)]


def done_images(csv_path: Path) -> Set[str]:
    """Images already listed in an existing results file."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {row["image"] for row in csv.DictReader(fh) if row.get("image")}


def existing_unit(csv_path: Path) -> Optional[str]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
    except Exception:
        return None
    for h in header:
        if h.startswith("lengths_"):
            return h.split("_", 1)[1]
    return None


def write_csv(csv_path: Path, results: Iterable[ImageResult], um_per_px: float) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(csv_header(um_per_px))
        for r in results:
            if not r.error:
                w.writerow(csv_row(r, um_per_px))


def annotated_path(folder: Path, rel: str) -> Path:
    p = folder / rel
    return p.with_name(p.stem + "_annotated.jpg")


def run_batch(engine: Engine, files: List[Path], root: Path, params: Params, um_per_px: float,
              csv_path: Path, append: bool = False, annotated_dir: Optional[Path] = None,
              on_result: Optional[Callable[[int, int, ImageResult], None]] = None,
              should_stop: Callable[[], bool] = lambda: False,
              log: Callable[[str], None] = print) -> List[ImageResult]:
    """Process files in order, appending one CSV row per image as soon as it is ready."""
    import cv2
    from .engine import load_rgb

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and csv_path.is_file() and csv_path.stat().st_size > 0 else "w"
    results: List[ImageResult] = []
    with csv_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(csv_header(um_per_px))
            fh.flush()
        n = len(files)
        for i, path in enumerate(files, 1):
            if should_stop():
                log("Stopped by the user")
                break
            res = engine.run_file(path, root, params)
            if res.error:
                log(f"{res.image}: {res.error}")
            else:
                writer.writerow(csv_row(res, um_per_px))
                fh.flush()
                if annotated_dir is not None:
                    rgb = load_rgb(path)
                    if rgb is not None:
                        out = annotated_path(annotated_dir, res.image)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        img = viz.draw(rgb, res, um_per_px)
                        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 92])
                        if ok:
                            buf.tofile(str(out))
            results.append(res)
            if on_result:
                on_result(i, n, res)
    return results
