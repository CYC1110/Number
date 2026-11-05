"""Preserved legacy single-color source; new CLI disables implicit caching."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class StimuliConfig:
    """Configuration for procedurally generated number stimuli."""

    H: int = 227
    W: int = 227
    number_sets: Tuple[int, ...] = (
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
    )
    repeats: int = 50
    base_r: float = 7.0
    jitter: float = 0.7
    set2_total_area: float = 1200.0
    set2_target_dist_range: Tuple[float, float] = (90.0, 100.0)
    set3_pentagon_radius: float = 110.0
    min_dist: float = 0.0
    border_margin: float = 0.0
    set3_distance_factor: float = math.sqrt(2.0)
    seed: int = 0
    cache_path: Optional[str] = None
    auto_cache: bool = False
    cache_dir: Optional[str] = None

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 0)

def _circle_mask(h: int, w: int, cx: int, cy: int, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)

def _rect_mask(h: int, w: int, cx: int, cy: int, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)

def _ellipse_mask(h: int, w: int, cx: int, cy: int, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2) / (rx * rx) + ((yy - cy) ** 2) / (ry * ry) <= 1.0

def _triangle_mask(h: int, w: int, cx: float, cy: float, r: float, ang: float) -> np.ndarray:
    verts = np.array(
        [
            [cx + r * math.cos(ang + k * 2 * np.pi / 3), cy + r * math.sin(ang + k * 2 * np.pi / 3)]
            for k in range(3)
        ]
    )
    minx, miny = np.floor(verts.min(0)).astype(int)
    maxx, maxy = np.ceil(verts.max(0)).astype(int)
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, w - 1)
    maxy = min(maxy, h - 1)
    if minx > maxx or miny > maxy:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[miny : maxy + 1, minx : maxx + 1]
    A, B, C = verts[0], verts[1], verts[2]
    v0 = B - A
    v1 = C - A
    den = v0[0] * v1[1] - v1[0] * v0[1] + 1e-12
    v2x = xx - A[0]
    v2y = yy - A[1]
    a = (v2x * v1[1] - v1[0] * v2y) / den
    b = (v0[0] * v2y - v2x * v0[1]) / den
    c = 1 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    mask = np.zeros((h, w), dtype=bool)
    mask[miny : maxy + 1, minx : maxx + 1] = inside
    return mask

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

def _place_nonoverlapping(
    h: int,
    w: int,
    radii: np.ndarray,
    rng: np.random.Generator,
    region_path: Optional[object] = None,
    distance_factor: float = 1.0,
    max_attempts_total: int = 4000,
    per_item_attempts: int = 300,
    min_distance: float = 0.0,
    border_margin: float = 0.0,
) -> List[Tuple[int, int, float]]:
    positions: List[Tuple[int, int, float]] = []
    if len(radii) == 0:
        return positions
    coords = np.empty((0, 2), dtype=np.float32)
    radii_existing = np.empty((0,), dtype=np.float32)
    attempts_total = 0
    for r in radii:
        if r <= 0:
            continue
        margin = max(int(math.ceil(r + border_margin)), 1)
        placed = False
        tries = 0
        while not placed and attempts_total < max_attempts_total and tries < per_item_attempts:
            x = int(rng.integers(margin, max(margin + 1, w - margin + 1)))
            y = int(rng.integers(margin, max(margin + 1, h - margin + 1)))
            if not _point_in_region(region_path, (x, y)):
                tries += 1
                attempts_total += 1
                continue
            if coords.shape[0]:
                dx = coords[:, 0] - x
                dy = coords[:, 1] - y
                min_allowed = distance_factor * (radii_existing + r) + min_distance
                if np.any(dx * dx + dy * dy < (min_allowed**2)):
                    tries += 1
                    attempts_total += 1
                    continue
            positions.append((x, y, float(r)))
            coords = np.vstack([coords, [x, y]]) if coords.size else np.array([[x, y]], dtype=np.float32)
            radii_existing = np.append(radii_existing, r)
            placed = True
            tries += 1
            attempts_total += 1
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

def generate_stimuli(cfg: StimuliConfig) -> Dict[str, np.ndarray]:
    """Generate number stimuli for all configured numerosities and sets."""

    cache_path = None  # Reference frontend deliberately disables implicit caches.

    rng = _rng(cfg.seed)
    H, W = cfg.H, cfg.W
    images: List[np.ndarray] = []
    set_id: List[int] = []
    numeros: List[int] = []

    for n in tqdm(cfg.number_sets):
        for _ in range(cfg.repeats):
            # Set 1: jittered circles
            img1 = np.zeros((H, W), dtype=np.float32)
            if n > 0:
                success = False
                attempt = 0
                while attempt < 50 and not success:
                    radii1 = cfg.base_r + cfg.jitter * rng.standard_normal(n)
                    radii1 = np.clip(radii1, 0.1, None)
                    pos1 = _place_nonoverlapping(
                        H,
                        W,
                        radii1,
                        rng,
                        min_distance=cfg.min_dist,
                        border_margin=cfg.border_margin,
                    )
                    if len(pos1) == n:
                        for (x, y, r) in pos1:
                            img1[_circle_mask(H, W, x, y, r)] = 1.0
                        success = True
                    attempt += 1
            if n > 0 and not success:
                raise RuntimeError("Set 1 placement failed")
            images.append(img1)
            set_id.append(1)
            numeros.append(n)

            # Set 2: constant total area + target mean distance
            img2 = np.zeros((H, W), dtype=np.float32)
            if n > 0:
                success = False
                attempt = 0
                while attempt < 50 and not success:
                    eps = rng.standard_normal(n)
                    radii2 = cfg.base_r + cfg.jitter * eps
                    radii2 = np.clip(radii2, 0.1, None)
                    area_sum = float(np.sum(np.pi * radii2**2))
                    scaling = math.sqrt(area_sum / cfg.set2_total_area)
                    radii2 = radii2 / scaling
                    average_dist = 0.0
                    distance_attempts = 0
                    pos2: List[Tuple[int, int, float]] = []
                    while (
                        (not pos2 or (n > 1 and not (
                            cfg.set2_target_dist_range[0]
                            <= average_dist
                            <= cfg.set2_target_dist_range[1]
                        )))
                        and distance_attempts < 100
                    ):
                        pos2 = _place_nonoverlapping(
                            H,
                            W,
                            radii2,
                            rng,
                            min_distance=cfg.min_dist,
                            border_margin=cfg.border_margin,
                        )
                        if len(pos2) == n:
                            average_dist = _mean_interpoint_distance(pos2)
                        else:
                            distance_attempts += 1
                            continue
                        distance_attempts += 1
                    if (
                        (n == 1 or cfg.set2_target_dist_range[0]
                        <= average_dist
                        <= cfg.set2_target_dist_range[1])
                        and len(pos2) == n
                    ):
                        for (x, y, r) in pos2:
                            img2[_circle_mask(H, W, x, y, r)] = 1.0
                        success = True
                    attempt += 1
            if n > 0 and not success:
                raise RuntimeError("Set 2 placement/distance control failed")
            images.append(img2)
            set_id.append(2)
            numeros.append(n)

            # Set 3: mixed shapes inside pentagon (sqrt(2) spacing)
            img3 = np.zeros((H, W), dtype=np.float32)
            if n > 0 and MplPath is not None:
                cx, cy = W // 2, H // 2
                theta0 = rng.uniform(0.0, 2 * np.pi)
                verts = [
                    (
                        cx + cfg.set3_pentagon_radius * math.cos(theta0 + i * 2 * np.pi / 5),
                        cy + cfg.set3_pentagon_radius * math.sin(theta0 + i * 2 * np.pi / 5),
                    )
                    for i in range(5)
                ]
                pent_path = _make_region_path(verts)
                radii3 = cfg.base_r + cfg.jitter * rng.standard_normal(n)
                radii3 = np.clip(radii3, 0.1, None)
                pos3 = _place_nonoverlapping(
                    H,
                    W,
                    radii3,
                    rng,
                    region_path=pent_path,
                    distance_factor=cfg.set3_distance_factor,
                    min_distance=cfg.min_dist,
                    border_margin=cfg.border_margin,
                )
                if len(pos3) != n:
                    raise RuntimeError("Set 3 placement failed")
                for (x, y, r) in pos3:
                    p = rng.random()
                    if p > 0.75:
                        mask = _rect_mask(H, W, x, y, r)
                    elif p > 0.5:
                        mask = _circle_mask(H, W, x, y, r)
                    elif p > 0.25:
                        mask = _ellipse_mask(H, W, x, y, r, 0.5 * r)
                    else:
                        ang = rng.uniform(0.0, 2 * np.pi)
                        mask = _triangle_mask(H, W, x, y, r, ang)
                    img3[mask] = 1.0
            images.append(img3)
            set_id.append(3)
            numeros.append(n)

    images_np = (np.clip(np.stack(images, axis=0), 0.0, 1.0) * 255.0).astype(np.uint8)
    set_id_np = np.asarray(set_id, dtype=np.int32)
    numeros_np = np.asarray(numeros, dtype=np.int32)

    meta = dict(
        H=H,
        W=W,
        seed=cfg.seed,
        number_sets=list(cfg.number_sets),
        repeats=cfg.repeats,
        base_r=cfg.base_r,
        jitter=cfg.jitter,
        set2_total_area=cfg.set2_total_area,
        set2_target_dist_range=cfg.set2_target_dist_range,
        set3_pentagon_radius=cfg.set3_pentagon_radius,
        min_dist=cfg.min_dist,
        border_margin=cfg.border_margin,
        set3_distance_factor=cfg.set3_distance_factor,
    )

    return {
        "images": images_np,
        "set_id": set_id_np,
        "numeros": numeros_np,
        "meta": meta,
    }
