"""Centerline length of an instance mask, in pixels.

The measurement follows the trichome post-processing used for the paper:
hole filling, tip dilation, skeleton longest path, smoothing, and merging
of collinear fragments.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _fill_holes(mask_u8: "np.ndarray") -> "np.ndarray":
    import cv2
    import numpy as np
    if mask_u8.max() == 0:
        return mask_u8
    h, w = mask_u8.shape[:2]
    flood = mask_u8.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return cv2.bitwise_or(mask_u8, cv2.bitwise_not(flood))


def _pca_angle(cnt: "np.ndarray") -> float:
    """Principal axis angle in degrees from PCA of contour points."""
    import numpy as np
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 3:
        return 0.0
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _anisotropic_dilate(mask_u8: "np.ndarray", cnt: "np.ndarray",
                        dilate_px: int) -> "np.ndarray":
    """Dilate mask along the principal axis of the contour to extend tips."""
    import cv2
    import numpy as np
    if dilate_px <= 0:
        return mask_u8.copy()
    angle = _pca_angle(cnt)
    h, w = mask_u8.shape[:2]
    pad = dilate_px + 10
    padded = cv2.copyMakeBorder(mask_u8, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape[:2]
    pcx, pcy = pw / 2.0, ph / 2.0
    M_rot = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
    rotated = cv2.warpAffine(padded, M_rot, (pw, ph), flags=cv2.INTER_NEAREST)
    kw = dilate_px * 2 + 1
    kh = max(3, dilate_px // 2 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    dilated = cv2.dilate(rotated, kernel, iterations=1)
    M_back = cv2.getRotationMatrix2D((pcx, pcy), -angle, 1.0)
    restored = cv2.warpAffine(dilated, M_back, (pw, ph), flags=cv2.INTER_NEAREST)
    return restored[pad: pad + h, pad: pad + w]


# ---------------------------------------------------------------------------
# Skeletonization & longest path
# ---------------------------------------------------------------------------

def _prune_skeleton(skel: "np.ndarray",
                    min_branch: int = 20) -> "np.ndarray":
    """Remove short branches from a binary skeleton (1/0 uint8).
    Uses convolution for fast neighbor counting — iteratively peels
    endpoint pixels min_branch times, which removes branches shorter
    than that many pixels."""
    import cv2
    import numpy as np
    s = skel.copy().astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(min_branch):
        nb = cv2.filter2D(s, cv2.CV_16U, kernel) - s.astype(np.uint16)
        endpoints = (s > 0) & (nb == 1)
        if not endpoints.any():
            break
        s[endpoints] = 0
    return s


def _skeleton_longest_path(
    mask_u8: "np.ndarray",
) -> Tuple[float, List[Tuple[float, float]]]:
    """Skeletonize, prune short branches, find longest path."""
    import cv2
    import numpy as np

    bin01 = (mask_u8 > 0).astype(np.uint8)
    if bin01.max() == 0:
        return 0.0, []
    try:
        skeleton = cv2.ximgproc.thinning(
            bin01 * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        skeleton = (skeleton > 0).astype(np.uint8)
    except AttributeError:
        skeleton = _thinning_fallback(bin01)

    skeleton = _prune_skeleton(skeleton)

    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return 0.0, []

    nodes = set(zip(xs.tolist(), ys.tolist()))
    dirs = [
        (-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),                            (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),  (0, 1, 1.0),  (1, 1, math.sqrt(2.0)),
    ]

    def dijkstra(start):
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float("inf")):
                continue
            for dx, dy, w in dirs:
                v = (u[0] + dx, u[1] + dy)
                if v not in nodes:
                    continue
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        far = max(dist.items(), key=lambda kv: kv[1])[0]
        return far, dist, prev

    start = (int(xs[0]), int(ys[0]))
    a, _, _ = dijkstra(start)
    b, dist_b, prev_b = dijkstra(a)
    length = dist_b.get(b, 0.0)

    path: List[Tuple[float, float]] = []
    cur = b
    path.append((float(cur[0]), float(cur[1])))
    while cur in prev_b:
        cur = prev_b[cur]
        path.append((float(cur[0]), float(cur[1])))
        if cur == a:
            break
    path.reverse()
    return float(length), path


def _thinning_fallback(bin01: "np.ndarray") -> "np.ndarray":
    """Zhang-Suen thinning fallback when cv2.ximgproc is unavailable."""
    import numpy as np
    img = (bin01 > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return img

    def _nb(y, x):
        return (img[y-1,x], img[y-1,x+1], img[y,x+1], img[y+1,x+1],
                img[y+1,x], img[y+1,x-1], img[y,x-1], img[y-1,x-1])

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_del = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] == 0:
                        continue
                    p = _nb(y, x)
                    bp = sum(p)
                    if bp < 2 or bp > 6:
                        continue
                    ap = sum(
                        p[i] == 0 and p[(i + 1) % 8] == 1 for i in range(8)
                    )
                    if ap != 1:
                        continue
                    if step == 0:
                        if p[0]*p[2]*p[4] != 0 or p[2]*p[4]*p[6] != 0:
                            continue
                    else:
                        if p[0]*p[2]*p[6] != 0 or p[0]*p[4]*p[6] != 0:
                            continue
                    to_del.append((y, x))
            if to_del:
                for y, x in to_del:
                    img[y, x] = 0
                changed = True
    return img


# ---------------------------------------------------------------------------
# Centerline smoothing, trimming & length
# ---------------------------------------------------------------------------

def _remove_interior_kinks(path: List[Tuple[float, float]],
                           window: int = 15,
                           min_angle: float = 110.0) -> List[Tuple[float, float]]:
    """Remove V-shaped kinks inside the path caused by skeleton branches.
    Detects sharp angles and bridges over them with a straight segment."""
    pts = list(path)
    for _ in range(10):
        if len(pts) < 2 * window + 3:
            break
        worst_i, worst_a = -1, 180.0
        for i in range(window, len(pts) - window):
            ax, ay = pts[i - window]
            bx, by = pts[i]
            cx, cy = pts[i + window]
            v1x, v1y = ax - bx, ay - by
            v2x, v2y = cx - bx, cy - by
            m1 = math.hypot(v1x, v1y)
            m2 = math.hypot(v2x, v2y)
            if m1 < 1e-9 or m2 < 1e-9:
                continue
            dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (m1 * m2)))
            angle = math.degrees(math.acos(dot))
            if angle < worst_a:
                worst_a = angle
                worst_i = i
        if worst_i < 0 or worst_a >= min_angle:
            break
        zone = int(window * 1.5)
        lo = max(0, worst_i - zone)
        hi = min(len(pts), worst_i + zone + 1)
        pts = pts[:lo] + pts[hi:]
    return pts


def _simplify_path(path: List[Tuple[float, float]],
                   epsilon: float = 8.0) -> List[Tuple[float, float]]:
    """Douglas-Peucker simplification + dense re-interpolation."""
    import cv2
    import numpy as np
    if len(path) < 5:
        return list(path)
    arr = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(arr, epsilon, False)
    key_pts = [(float(p[0][0]), float(p[0][1])) for p in simplified]
    if len(key_pts) < 2:
        return list(path)
    dense: List[Tuple[float, float]] = [key_pts[0]]
    for i in range(1, len(key_pts)):
        x0, y0 = key_pts[i - 1]
        x1, y1 = key_pts[i]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n_steps = max(1, int(round(seg_len)))
        for s in range(1, n_steps + 1):
            t = s / n_steps
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense


def _smooth_path(path: List[Tuple[float, float]],
                 sigma: float = 6.0) -> List[Tuple[float, float]]:
    """Gaussian-smooth the centerline coords independently."""
    import numpy as np
    if len(path) < 5:
        return list(path)
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    xs = np.array([p[0] for p in path], dtype=np.float64)
    ys = np.array([p[1] for p in path], dtype=np.float64)
    xs = gaussian_filter1d(xs, sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(ys, sigma=sigma, mode="nearest")
    return [(float(xs[i]), float(ys[i])) for i in range(len(xs))]


def _trim_sharp_ends(path: List[Tuple[float, float]],
                     max_angle_deg: float = 60.0,
                     check_pts: int = 12) -> List[Tuple[float, float]]:
    """Remove points at the beginning/end where the path makes a sharp bend."""
    if len(path) < 6:
        return list(path)

    def _angle_at(pts: Sequence[Tuple[float, float]], i: int) -> float:
        ax, ay = pts[i - 1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = cx - bx, cy - by
        m1 = math.hypot(v1x, v1y)
        m2 = math.hypot(v2x, v2y)
        if m1 < 1e-9 or m2 < 1e-9:
            return 180.0
        dot = (v1x * v2x + v1y * v2y) / (m1 * m2)
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    start = 0
    for i in range(1, min(check_pts, len(path) - 1)):
        ang = _angle_at(path, i)
        if ang < max_angle_deg:
            start = i + 1
            break

    end = len(path)
    for i in range(len(path) - 2, max(start, len(path) - 1 - check_pts), -1):
        ang = _angle_at(path, i)
        if ang < max_angle_deg:
            end = i
            break

    trimmed = path[start:end]
    return trimmed if len(trimmed) >= 2 else list(path)


def _trim_uturn(path: List[Tuple[float, float]],
                max_ratio: float = 0.25) -> List[Tuple[float, float]]:
    """Detect U-turns: if endpoint distance / path length < max_ratio,
    the path doubles back. Cut at the turnaround (farthest from start)."""
    if len(path) < 10:
        return list(path)

    total_len = _polyline_length(path)
    if total_len < 20:
        return list(path)

    ep_dist = math.hypot(path[-1][0] - path[0][0],
                         path[-1][1] - path[0][1])
    ratio = ep_dist / total_len

    if ratio >= max_ratio:
        return list(path)

    sx, sy = path[0]
    best_dist = 0.0
    best_idx = 0
    for i, (px, py) in enumerate(path):
        d = math.hypot(px - sx, py - sy)
        if d > best_dist:
            best_dist = d
            best_idx = i

    result = path[:best_idx + 1]
    return result if len(result) >= 2 else list(path)


def _polyline_length(path: List[Tuple[float, float]]) -> float:
    """Euclidean length along a polyline."""
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(path[i + 1][0] - path[i][0],
                            path[i + 1][1] - path[i][1])
    return total


# ---------------------------------------------------------------------------
# Instance data structure
# ---------------------------------------------------------------------------

@dataclass
class _Instance:
    mask: "np.ndarray"
    contour: "np.ndarray"
    path: List[Tuple[float, float]]
    length: float
    merged: bool = False


# ---------------------------------------------------------------------------
# Centerline merging
# ---------------------------------------------------------------------------

def _tangent_vec(path: List[Tuple[float, float]], at_start: bool,
                 n_pts: int = 15) -> Tuple[float, float]:
    """Unit tangent vector at one end of the path."""
    if len(path) < 2:
        return (0.0, 0.0)
    n = min(n_pts, len(path) - 1)
    if at_start:
        dx = path[n][0] - path[0][0]
        dy = path[n][1] - path[0][1]
    else:
        dx = path[-1][0] - path[-1 - n][0]
        dy = path[-1][1] - path[-1 - n][1]
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def _endpoint_dist(p1: Tuple[float, float],
                   p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle_between(v1: Tuple[float, float],
                   v2: Tuple[float, float]) -> float:
    """Angle in degrees between two unit vectors (0..180)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _junction_deviation(cand: List[Tuple[float, float]],
                        n_a: int, n_pts: int = 15) -> float:
    """Perpendicular distance of the junction points from the chord
    connecting deeper points on each side.  Catches lateral jumps
    even when path directions are parallel."""
    k = min(n_pts, n_a - 1, len(cand) - n_a - 1)
    if k < 2:
        return 0.0
    ax, ay = cand[n_a - 1 - k]
    bx, by = cand[n_a + k]
    chord_dx = bx - ax
    chord_dy = by - ay
    chord_len = math.hypot(chord_dx, chord_dy)
    if chord_len < 1e-6:
        return 0.0
    nx = -chord_dy / chord_len
    ny = chord_dx / chord_len
    max_dev = 0.0
    for idx in (n_a - 1, n_a):
        dx = cand[idx][0] - ax
        dy = cand[idx][1] - ay
        max_dev = max(max_dev, abs(dx * nx + dy * ny))
    return max_dev


def _junction_turn(cand: List[Tuple[float, float]],
                   n_a: int, n_pts: int = 15) -> float:
    """Direction change between path A and path B at the junction.
    0° = same direction, 180° = U-turn."""
    k = min(n_pts, n_a - 1, len(cand) - n_a - 1)
    if k < 2:
        return 0.0
    a_dx = cand[n_a - 1][0] - cand[n_a - 1 - k][0]
    a_dy = cand[n_a - 1][1] - cand[n_a - 1 - k][1]
    b_dx = cand[n_a + k][0] - cand[n_a][0]
    b_dy = cand[n_a + k][1] - cand[n_a][1]
    ma = math.hypot(a_dx, a_dy)
    mb = math.hypot(b_dx, b_dy)
    if ma < 1e-6 or mb < 1e-6:
        return 0.0
    dot = (a_dx * b_dx + a_dy * b_dy) / (ma * mb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _try_merge(a: _Instance, b: _Instance,
               max_dist: float, max_angle: float,
               max_turn: float = 60.0) -> bool:
    """Merge b into a if their centerlines form a smooth continuation.

    Checks:
      1. Endpoint distance <= max_dist
      2. Tangent alignment at the connected endpoints
      3. Direction change at the junction <= max_turn degrees
    """
    import cv2

    if a.merged or b.merged:
        return False
    pa, pb = a.path, b.path
    if len(pa) < 2 or len(pb) < 2:
        return False

    best = None  # (turn_angle, cand, n_a)

    combos = [
        (pa[-1], pb[0],  False, False),
        (pa[-1], pb[-1], False, True),
        (pa[0],  pb[0],  True,  False),
        (pa[0],  pb[-1], True,  True),
    ]

    for ep_a, ep_b, a_is_start, b_is_end in combos:
        d = _endpoint_dist(ep_a, ep_b)
        if d > max_dist:
            continue

        tv_a = _tangent_vec(pa, at_start=a_is_start)
        tv_b = _tangent_vec(pb, at_start=b_is_end)
        if tv_a == (0.0, 0.0) or tv_b == (0.0, 0.0):
            continue

        angle = _angle_between(tv_a, tv_b)
        aligned = angle < max_angle or angle > (180.0 - max_angle)
        if not aligned:
            continue

        if a_is_start:
            cand = list(reversed(pa))
        else:
            cand = list(pa)
        n_a = len(cand)
        if b_is_end:
            cand.extend(reversed(pb))
        else:
            cand.extend(pb)

        turn = _junction_turn(cand, n_a)
        if turn > max_turn:
            continue

        dev = _junction_deviation(cand, n_a)
        if dev > 5.0:
            continue

        if best is None or turn < best[0]:
            best = (turn, cand, n_a)

    if best is None:
        return False

    _, cand, _ = best
    a.path = cand
    a.length = a.length + b.length
    a.mask = cv2.bitwise_or(a.mask, b.mask)
    cnts, _ = cv2.findContours(a.mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        a.contour = max(cnts, key=cv2.contourArea)
    b.merged = True
    return True


def _dedup_overlapping(instances: List[_Instance],
                       overlap_thresh: float = 0.35) -> List[_Instance]:
    """If two masks overlap significantly (the smaller mask has >overlap_thresh
    of its area covered by the larger), discard the shorter centerline."""
    import cv2
    import numpy as np

    n = len(instances)
    areas = [int(inst.mask.sum()) // 255 for inst in instances]

    for i in range(n):
        if instances[i].merged:
            continue
        for j in range(i + 1, n):
            if instances[j].merged:
                continue
            overlap = int(cv2.bitwise_and(instances[i].mask, instances[j].mask).sum()) // 255
            if overlap == 0:
                continue
            smaller_area = min(areas[i], areas[j])
            if smaller_area == 0:
                continue
            if overlap / smaller_area >= overlap_thresh:
                if instances[i].length >= instances[j].length:
                    instances[j].merged = True
                else:
                    instances[i].merged = True

    return [inst for inst in instances if not inst.merged]


def _merge_instances(instances: List[_Instance],
                     max_dist: float, max_angle: float,
                     max_turn: float = 60.0,
                     smooth_sigma: float = 6.0,
                     trim_angle_deg: float = 60.0) -> List[_Instance]:
    """Greedily merge instance pairs whose centerlines are continuations.
    After merging, re-trim / re-smooth / re-measure.
    Then deduplicate overlapping masks (parallel split artifacts)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(instances)):
            if instances[i].merged:
                continue
            for j in range(i + 1, len(instances)):
                if instances[j].merged:
                    continue
                if _try_merge(instances[i], instances[j],
                              max_dist, max_angle, max_turn):
                    changed = True

    # deduplicate parallel overlapping masks (YOLO split artifacts)
    instances = _dedup_overlapping(instances)

    result = instances

    for inst in result:
        if len(inst.path) >= 5:
            inst.path = _trim_uturn(inst.path)
        if len(inst.path) >= 5:
            inst.path = _remove_interior_kinks(inst.path)
        if len(inst.path) >= 5:
            inst.path = _simplify_path(inst.path)
        if len(inst.path) >= 5:
            inst.path = _smooth_path(inst.path, sigma=smooth_sigma)
        if len(inst.path) >= 5:
            inst.path = _trim_sharp_ends(inst.path, max_angle_deg=trim_angle_deg)
        inst.length = _polyline_length(inst.path)

    return result


# ---------------------------------------------------------------------------
# Centerline computation for one instance
# ---------------------------------------------------------------------------

def _compute_centerline(
    mask: "np.ndarray",
    cnt: "np.ndarray",
    contour_dilate_px: int,
    aniso_dilate_px: int,
    smooth_sigma: float = 6.0,
    trim_angle_deg: float = 60.0,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Compute centerline on a dilated local crop of the mask."""
    import cv2
    import numpy as np

    h, w = mask.shape[:2]
    x0, y0, xw, yh = cv2.boundingRect(cnt)
    pad = max(contour_dilate_px, aniso_dilate_px) + 15
    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(w, x0 + xw + pad)
    ry1 = min(h, y0 + yh + pad)

    local_mask = mask[ry0:ry1, rx0:rx1].copy()
    local_cnt = cnt.copy()
    local_cnt[:, 0, 0] -= rx0
    local_cnt[:, 0, 1] -= ry0

    if contour_dilate_px > 0:
        k = contour_dilate_px * 2 + 1
        iso_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        local_mask = cv2.dilate(local_mask, iso_kernel, iterations=1)

    dilated = _anisotropic_dilate(local_mask, local_cnt, aniso_dilate_px)

    _, path = _skeleton_longest_path(dilated)

    path = [(x + rx0, y + ry0) for x, y in path]

    if len(path) >= 5:
        path = _trim_uturn(path)
    if len(path) >= 5:
        path = _remove_interior_kinks(path)
    if len(path) >= 5:
        path = _simplify_path(path)
    if len(path) >= 5:
        path = _smooth_path(path, sigma=smooth_sigma)
    if len(path) >= 5:
        path = _trim_sharp_ends(path, max_angle_deg=trim_angle_deg)

    length = _polyline_length(path)
    return length, path
