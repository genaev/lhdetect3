"""Colors and drawing of predicted trichomes."""
from __future__ import annotations

import colorsys
from typing import List, Tuple

import numpy as np

_GOLDEN = 0.618033988749895


def palette(n: int) -> List[Tuple[int, int, int]]:
    """n distinct bright colors (RGB) with hues spread by the golden ratio."""
    out = []
    h = 0.11
    for i in range(n):
        s = 0.95 if i % 2 == 0 else 0.75
        v = 1.0 if i % 3 != 2 else 0.85
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
        h += _GOLDEN
    return out


def draw(rgb: np.ndarray, res, um_per_px: float = 0.0, contours: bool = True,
         centerlines: bool = True, numbers: bool = True) -> np.ndarray:
    """RGB image with the contour of every trichome in its own color."""
    import cv2
    img = rgb.copy()
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 500)))
    font = max(0.5, min(h, w) / 1400)
    colors = palette(len(res.trichomes))
    for i, (t, c) in enumerate(zip(res.trichomes, colors), 1):
        if contours and t.contour is not None:
            cv2.polylines(img, [t.contour.reshape(-1, 1, 2)], True, c, thick, cv2.LINE_AA)
        elif contours:   # detected, but the mask is empty: show the box
            x1, y1, x2, y2 = (int(round(v)) for v in t.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, max(1, thick // 2), cv2.LINE_AA)
        if centerlines and len(t.path) > 1:
            pts = np.round(np.array(t.path)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (255, 255, 255), max(1, thick // 2 + 1), cv2.LINE_AA)
            cv2.polylines(img, [pts], False, c, max(1, thick // 2), cv2.LINE_AA)
        if numbers:
            if t.path:
                x, y = t.path[len(t.path) // 2]
            elif t.contour is not None:
                x, y = t.contour.mean(0)
            else:
                x, y = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            org = (int(x) + thick * 2, int(y))
            cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, font, c, max(1, thick - 1), cv2.LINE_AA)
    return img
