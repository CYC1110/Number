"""Preserved dot drawing and placement primitives; no CLI or side effects."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict
import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class StimuliConfig:
    """Configuration for white/black dot stimuli."""

    H: int = 227
    W: int = 227
    number_sets: Tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    )
    repeats: int = 200
    per_combo_repeats: Optional[int] = None
    w_values: Optional[Tuple[int, ...]] = None
    b_values: Optional[Tuple[int, ...]] = None
    base_r: float = 7.0
    jitter: float = 0.7
    set2_total_area: float = 1200.0
    set2_split_total_area_by_color: bool = True
    set2_target_dist_range: Tuple[float, float] = (90.0, 100.0)
    set3_pentagon_radius: float = 110.0
    min_dist: float = 1.0
    border_margin: float = 2.0
    set3_distance_factor: float = math.sqrt(2.0)
    seed: int = 0
    background_gray: int = 127
    white_value: int = 255
    black_value: int = 0

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 0)

def _circle_mask(h: int, w: int, cx: int, cy: int, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)

def _make_region_path(verts: Sequence[Tuple[float, float]]) -> Optional[object]:
    if MplPath is None:
        return None
    return MplPath(verts)

def _point_in_region(region_path: Optional[object], pt: Tuple[float, float]) -> bool:
    if region_path is None:
        return True
    try:
        return bool(region_path.contains_point(pt))
    except Exception:  # pragma: no cover - robustness for Matplotlib quirks
        return False

def _dilate_mask_square(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a boolean mask with a 3x3 square structuring element."""
    if iterations <= 0:
        return mask.copy()
    out = mask.astype(bool, copy=True)
    for _ in range(int(iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = expanded
    return out

def _place_nonoverlapping(
    h: int,
    w: int,
    radii: np.ndarray,
    rng: np.random.Generator,
    region_path: Optional[object] = None,
    distance_factor: float = 1.0,
    max_attempts_total: int = 12000,
    per_item_attempts: int = 800,
    min_distance: float = 0.0,
    border_margin: float = 0.0,
    raster_gap: Optional[int] = None,
    return_stats: bool = False,
):
    positions: List[Tuple[int, int, float]] = []
    positive_radii = np.asarray(radii, dtype=np.float32)
    target_count = int(np.count_nonzero(positive_radii > 0))
    positions_by_index: List[Optional[Tuple[int, int, float]]] = [None] * len(positive_radii)
    stats = {
        "target_dot_count": target_count,
        "placed_dot_count": 0,
        "candidate_attempts": 0,
        "candidate_accepts": 0,
        "candidate_rejections": 0,
        "reject_region": 0,
        "reject_center_distance": 0,
        "reject_raster_overlap": 0,
        "failed_items": 0,
        "hit_total_limit": 0,
        "hit_per_item_limit": 0,
        "max_attempts_total": int(max_attempts_total),
        "per_item_attempts": int(per_item_attempts),
    }
    if len(radii) == 0:
        stats["placement_success"] = True
        return (positions, stats) if return_stats else positions
    coords = np.empty((0, 2), dtype=np.float32)
    radii_existing = np.empty((0,), dtype=np.float32)
    gap_pixels = max(0, int(math.ceil(float(min_distance)))) if raster_gap is None else max(0, int(raster_gap))
    occupied_mask = np.zeros((h, w), dtype=bool)
    blocked_mask = occupied_mask.copy()
    attempts_total = 0
    placement_order = np.argsort(-positive_radii, kind="stable")
    for radius_index in placement_order:
        r = positive_radii[int(radius_index)]
        if r <= 0:
            continue
        margin = max(int(math.ceil(r + border_margin)), 1)
        placed = False
        tries = 0
        while not placed and attempts_total < max_attempts_total and tries < per_item_attempts:
            x = int(rng.integers(margin, max(margin + 1, w - margin + 1)))
            y = int(rng.integers(margin, max(margin + 1, h - margin + 1)))
            tries += 1
            attempts_total += 1
            stats["candidate_attempts"] += 1
            if not _point_in_region(region_path, (x, y)):
                stats["reject_region"] += 1
                continue
            if coords.shape[0]:
                dx = coords[:, 0] - x
                dy = coords[:, 1] - y
                min_allowed = distance_factor * (radii_existing + r) + min_distance
                if np.any(dx * dx + dy * dy < (min_allowed**2)):
                    stats["reject_center_distance"] += 1
                    continue
            candidate_mask = _circle_mask(h, w, x, y, float(r))
            if np.any(candidate_mask & blocked_mask):
                stats["reject_raster_overlap"] += 1
                continue
            positions_by_index[int(radius_index)] = (x, y, float(r))
            coords = np.vstack([coords, [x, y]]) if coords.size else np.array([[x, y]], dtype=np.float32)
            radii_existing = np.append(radii_existing, r)
            occupied_mask |= candidate_mask
            blocked_mask = _dilate_mask_square(occupied_mask, gap_pixels)
            placed = True
            stats["candidate_accepts"] += 1
        if not placed:
            stats["failed_items"] += 1
            if attempts_total >= max_attempts_total:
                stats["hit_total_limit"] = 1
            if tries >= per_item_attempts:
                stats["hit_per_item_limit"] += 1
    positions = [pos for pos in positions_by_index if pos is not None]
    stats["placed_dot_count"] = len(positions)
    stats["candidate_rejections"] = (
        int(stats["reject_region"]) + int(stats["reject_center_distance"]) + int(stats["reject_raster_overlap"])
    )
    stats["placement_success"] = len(positions) == target_count
    if return_stats:
        return positions, stats
    return positions

def _mean_interpoint_distance(positions: List[Tuple[int, int, float]]) -> float:
    n = len(positions)
    if n <= 1:
        return 95.0
    pts = np.array([(x, y) for x, y, _ in positions], dtype=np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.sqrt((diff**2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(dmat[iu].mean())

def _valid_combos(total: int, allowed_w: Sequence[int], allowed_b: Sequence[int]) -> List[Tuple[int, int]]:
    allowed_b_set = set(allowed_b)
    combos = [(w, total - w) for w in allowed_w if (total - w) in allowed_b_set]
    return sorted(combos, key=lambda pair: (pair[0], pair[1]))

def _combo_counts(
    rng: np.random.Generator,
    total_images: int,
    num_combos: int,
    per_combo_repeats: Optional[int] = None,
) -> List[int]:
    if per_combo_repeats is not None:
        return [int(per_combo_repeats) for _ in range(num_combos)]
    base = total_images // num_combos
    remainder = total_images % num_combos
    counts = [base for _ in range(num_combos)]
    if remainder > 0:
        extra_idx = rng.choice(num_combos, size=remainder, replace=False)
        for idx in extra_idx:
            counts[int(idx)] += 1
    return counts

def _assign_colors(n: int, w: int, rng: np.random.Generator) -> np.ndarray:
    colors = np.zeros(n, dtype=np.uint8)
    if w > 0:
        white_idx = rng.choice(n, size=w, replace=False)
        colors[white_idx] = 1
    return colors

def _draw_dots(
    h: int,
    w: int,
    positions: List[Tuple[int, int, float]],
    colors: np.ndarray,
    cfg: StimuliConfig,
) -> np.ndarray:
    img = np.full((h, w), cfg.background_gray, dtype=np.uint8)
    for (x, y, r), color_flag in zip(positions, colors):
        mask = _circle_mask(h, w, x, y, r)
        img[mask] = cfg.white_value if int(color_flag) == 1 else cfg.black_value
    return img
