"""Centerline length of trichome instance masks, in pixels.

For one instance mask:
  1. every connected piece of the mask is closed with a 3 x 3 kernel and thinned
     (Zhang-Suen); the longest path through its skeleton is its centerline;
  2. the path is smoothed with a Gaussian (sigma = 3 px) keeping its end points;
  3. both ends are prolonged along the local direction up to the mask edge,
     because a skeleton stops about half a width before the tip and the base;
  4. pieces of one mask that continue each other (gap <= 60 px, direction
     change <= 35 degrees) are joined, the gap included;
  5. the length is the length of the resulting polyline.

Only NumPy and OpenCV are required.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

SIGMA = 3.0
TANGENT_POINTS = 8
JOIN_DIST = 60.0
JOIN_ANGLE = 35.0
MIN_PART_AREA = 1


# --------------------------------------------------------------------------- basic steps

def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    import cv2
    if mask_u8.max() == 0:
        return mask_u8
    h, w = mask_u8.shape[:2]
    flood = mask_u8.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return cv2.bitwise_or(mask_u8, cv2.bitwise_not(flood))


def zhang_suen(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a binary image (vectorized)."""
    s = np.pad((img > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2 = s[:-2, 1:-1]; p3 = s[:-2, 2:]; p4 = s[1:-1, 2:]; p5 = s[2:, 2:]
            p6 = s[2:, 1:-1]; p7 = s[2:, :-2]; p8 = s[1:-1, :-2]; p9 = s[:-2, :-2]
            nb = (p2, p3, p4, p5, p6, p7, p8, p9)
            b = sum(x.astype(np.int16) for x in nb)
            a = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            d = (s[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if d.any():
                s[1:-1, 1:-1][d] = 0
                changed = True
        if not changed:
            return s[1:-1, 1:-1]


_STEPS = [(-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (1, 0, 1.0),
          (-1, 1, math.sqrt(2.0)), (0, 1, 1.0), (1, 1, math.sqrt(2.0))]


def longest_path(skel: np.ndarray) -> List[Point]:
    """Longest shortest path through an 8-connected skeleton (two Dijkstra sweeps)."""
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return []
    if len(xs) == 1:
        return [(float(xs[0]), float(ys[0]))]
    nodes = set(zip(xs.tolist(), ys.tolist()))

    def sweep(start):
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for dx, dy, w in _STEPS:
                v = (u[0] + dx, u[1] + dy)
                if v in nodes:
                    nd = d + w
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
        return max(dist, key=dist.get), prev

    a, _ = sweep((int(xs[0]), int(ys[0])))
    b, prev = sweep(a)
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return [(float(x), float(y)) for x, y in reversed(path)]


def polyline_length(path: Sequence[Point]) -> float:
    if len(path) < 2:
        return 0.0
    a = np.asarray(path, dtype=np.float64)
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def smooth_path(path: List[Point], sigma: float = SIGMA) -> List[Point]:
    """Gaussian smoothing of the coordinates; the end points stay in place."""
    if sigma <= 0 or len(path) < 5:
        return list(path)
    a = np.asarray(path, dtype=np.float64)
    r = int(4 * sigma + 0.5)
    t = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (t / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="reflect", reflect_type="odd")
    out = np.stack([np.convolve(pad[:, k], w, mode="valid") for k in range(2)], 1)
    out[0], out[-1] = a[0], a[-1]
    return [(float(x), float(y)) for x, y in out]


def extend_to_edge(path: List[Point], mask: np.ndarray, back: int = TANGENT_POINTS,
                   step: float = 0.25, max_ext: float = 80.0) -> List[Point]:
    """Prolong both ends along the local direction while staying inside the mask."""
    if len(path) < 2:
        return list(path)
    h, w = mask.shape

    def walk(end: Point, ref: Point) -> Optional[Point]:
        dx, dy = end[0] - ref[0], end[1] - ref[1]
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        dx, dy = dx / n, dy / n
        x, y = end
        last = None
        for _ in range(int(max_ext / step)):
            x += step * dx
            y += step * dy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h) or mask[yi, xi] == 0:
                break
            last = (x, y)
        return last

    k = min(back, len(path) - 1)
    head = walk(path[0], path[k])
    tail = walk(path[-1], path[-1 - k])
    return ([head] if head else []) + list(path) + ([tail] if tail else [])


def _direction(path: Sequence[Point], at_end: bool, k: int = 10) -> Point:
    k = min(k, len(path) - 1)
    a, b = (path[-1 - k], path[-1]) if at_end else (path[k], path[0])
    d = (b[0] - a[0], b[1] - a[1])
    n = math.hypot(*d) or 1.0
    return (d[0] / n, d[1] / n)


def join_paths(paths: List[List[Point]], max_dist: float = JOIN_DIST,
               max_angle: float = JOIN_ANGLE, align_both: bool = False) -> Tuple[List[Point], List[int]]:
    """Start from the longest path and attach the others that continue it.

    The gap must point along the end of the growing path; with align_both, the
    attached path must also run in the same direction. Returns the joined path
    and the indices of the attached paths."""
    order = sorted(range(len(paths)), key=lambda i: polyline_length(paths[i]), reverse=True)
    path = list(paths[order[0]])
    used = [order[0]]
    rest = [i for i in order[1:] if len(paths[i]) >= 2]
    cos_max = math.cos(math.radians(max_angle))
    changed = True
    while changed and rest and len(path) >= 2:
        changed = False
        for i in list(rest):
            best = None
            for q in (paths[i], paths[i][::-1]):
                for at_end in (True, False):
                    e = path[-1] if at_end else path[0]
                    s = q[0] if at_end else q[-1]
                    gap = (s[0] - e[0], s[1] - e[1])
                    g = math.hypot(*gap)
                    if g > max_dist:
                        continue
                    t = _direction(path, at_end)
                    if g > 1e-6 and (t[0] * gap[0] + t[1] * gap[1]) / g < cos_max:
                        continue
                    if align_both:
                        tq = _direction(q, at_end=not at_end)
                        if -(t[0] * tq[0] + t[1] * tq[1]) < cos_max:
                            continue
                    if best is None or g < best[0]:
                        best = (g, q, at_end)
            if best is not None:
                _, q, at_end = best
                path = path + list(q) if at_end else list(q) + path
                rest.remove(i)
                used.append(i)
                changed = True
    return path, used


# --------------------------------------------------------------------------- one mask

def measure_mask(mask: np.ndarray, sigma: float = SIGMA, min_part_area: int = MIN_PART_AREA,
                 join_dist: float = JOIN_DIST, join_angle: float = JOIN_ANGLE) -> Tuple[float, List[Point]]:
    """Centerline length (px) and path (image coordinates) of one instance mask."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    pts = cv2.findNonZero(m)
    if pts is None:
        return 0.0, []
    x0, y0, bw, bh = cv2.boundingRect(pts)
    pad = 2
    crop = np.zeros((bh + 2 * pad, bw + 2 * pad), np.uint8)
    crop[pad:pad + bh, pad:pad + bw] = m[y0:y0 + bh, x0:x0 + bw]
    n, lab, st, _ = cv2.connectedComponentsWithStats(crop, 8)
    kernel = np.ones((3, 3), np.uint8)
    parts: List[List[Point]] = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_part_area:
            continue
        x, y, w, h = st[i, :4]
        piece = np.pad((lab[y:y + h, x:x + w] == i).astype(np.uint8), 2)
        piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, kernel)
        p = longest_path(zhang_suen(piece))
        p = [(px + x - 2, py + y - 2) for px, py in p]
        p = smooth_path(p, sigma)
        p = extend_to_edge(p, crop)
        if p:
            parts.append(p)
    if not parts:
        return 0.0, []
    path, _ = join_paths(parts, join_dist, join_angle)
    path = [(px + x0 - pad, py + y0 - pad) for px, py in path]
    return polyline_length(path), path
