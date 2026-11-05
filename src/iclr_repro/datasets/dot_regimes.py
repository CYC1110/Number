"""Preserved constant-size, color-area and spatial-extent regimes."""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, List, Dict
import numpy as np
from .dot_core import MplPath, StimuliConfig, _assign_colors, _combo_counts, _draw_dots, _make_region_path, _mean_interpoint_distance, _place_nonoverlapping, _rng, _valid_combos


def _target_areas_for_color_split(colors: np.ndarray, cfg: StimuliConfig) -> Tuple[float, float]:
    n_white = int(np.count_nonzero(colors))
    n_black = int(colors.shape[0] - n_white)
    if n_white > 0 and n_black > 0:
        return float(cfg.set2_total_area * 0.5), float(cfg.set2_total_area * 0.5)
    if n_white > 0:
        return float(cfg.set2_total_area), 0.0
    if n_black > 0:
        return 0.0, float(cfg.set2_total_area)
    return 0.0, 0.0

def _sample_radii_from_uniform_area_budget(
    count: int,
    target_area: float,
    rng: np.random.Generator,
    min_area_per_dot: float,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    min_total = float(min_area_per_dot) * int(count)
    if target_area + 1e-9 < min_total:
        raise ValueError(
            f"target_area={target_area:.3f} is too small for count={count} "
            f"and min_area_per_dot={min_area_per_dot:.3f}"
        )
    remaining = max(float(target_area) - min_total, 0.0)
    weights = rng.uniform(0.0, 1.0, size=int(count)).astype(np.float64, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        weights = np.full((int(count),), 1.0 / int(count), dtype=np.float64)
    else:
        weights = weights / weight_sum
    areas = float(min_area_per_dot) + remaining * weights
    return np.sqrt(areas / math.pi).astype(np.float32, copy=False)

def _new_generation_stats() -> Dict[str, object]:
    return {
        "generation_success": False,
        "target_dot_count": 0,
        "placed_dot_count": 0,
        "radius_sample_attempts": 0,
        "layout_attempts": 0,
        "placement_call_count": 0,
        "placement_candidate_attempts": 0,
        "placement_candidate_accepts": 0,
        "placement_candidate_rejections": 0,
        "placement_reject_region": 0,
        "placement_reject_center_distance": 0,
        "placement_reject_raster_overlap": 0,
        "placement_failed_items": 0,
        "placement_hit_total_limit": 0,
        "placement_hit_per_item_limit": 0,
        "layout_reject_incomplete": 0,
        "layout_reject_distance": 0,
        "success_mean_interpoint_distance": float("nan"),
    }

def _merge_placement_stats(stats: Dict[str, object], placement_stats: Dict[str, object]) -> None:
    stats["placement_call_count"] = int(stats["placement_call_count"]) + 1
    stats["target_dot_count"] = max(int(stats["target_dot_count"]), int(placement_stats.get("target_dot_count", 0)))
    stats["placed_dot_count"] = max(int(stats["placed_dot_count"]), int(placement_stats.get("placed_dot_count", 0)))
    stats["placement_candidate_attempts"] = int(stats["placement_candidate_attempts"]) + int(
        placement_stats.get("candidate_attempts", 0)
    )
    stats["placement_candidate_accepts"] = int(stats["placement_candidate_accepts"]) + int(
        placement_stats.get("candidate_accepts", 0)
    )
    stats["placement_candidate_rejections"] = int(stats["placement_candidate_rejections"]) + int(
        placement_stats.get("candidate_rejections", 0)
    )
    stats["placement_reject_region"] = int(stats["placement_reject_region"]) + int(placement_stats.get("reject_region", 0))
    stats["placement_reject_center_distance"] = int(stats["placement_reject_center_distance"]) + int(
        placement_stats.get("reject_center_distance", 0)
    )
    stats["placement_reject_raster_overlap"] = int(stats["placement_reject_raster_overlap"]) + int(
        placement_stats.get("reject_raster_overlap", 0)
    )
    stats["placement_failed_items"] = int(stats["placement_failed_items"]) + int(placement_stats.get("failed_items", 0))
    stats["placement_hit_total_limit"] = int(stats["placement_hit_total_limit"]) + int(
        placement_stats.get("hit_total_limit", 0)
    )
    stats["placement_hit_per_item_limit"] = int(stats["placement_hit_per_item_limit"]) + int(
        placement_stats.get("hit_per_item_limit", 0)
    )

def _finalize_generation_stats(stats: Dict[str, object], *, success: bool, mean_distance: float = float("nan")) -> Dict[str, object]:
    stats["generation_success"] = bool(success)
    if success:
        stats["success_mean_interpoint_distance"] = float(mean_distance)
    radius_sample_retries = max(int(stats["radius_sample_attempts"]) - 1, 0)
    layout_retries = max(int(stats["layout_attempts"]) - 1, 0)
    candidate_retries = int(stats["placement_candidate_rejections"])
    stats["radius_sample_retries"] = int(radius_sample_retries)
    stats["layout_retries"] = int(layout_retries)
    stats["placement_candidate_retries"] = int(candidate_retries)
    stats["total_retry_count"] = int(radius_sample_retries + layout_retries + candidate_retries)
    return stats

def _generate_set1_positions_tracked(
    n: int,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    *,
    layout_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    stats = _new_generation_stats()
    stats["target_dot_count"] = int(n)
    if n == 0:
        return [], _finalize_generation_stats(stats, success=True, mean_distance=_mean_interpoint_distance([]))
    radii = cfg.base_r + cfg.jitter * rng.standard_normal(n)
    radii = np.clip(radii, 0.1, None)
    for _ in range(int(layout_attempts)):
        stats["layout_attempts"] = int(stats["layout_attempts"]) + 1
        pos, placement_stats = _place_nonoverlapping(
            cfg.H,
            cfg.W,
            radii,
            rng,
            max_attempts_total=int(placement_max_attempts_total),
            per_item_attempts=int(placement_per_item_attempts),
            min_distance=cfg.min_dist,
            border_margin=cfg.border_margin,
            return_stats=True,
        )
        _merge_placement_stats(stats, placement_stats)
        if len(pos) == n:
            mean_distance = _mean_interpoint_distance(pos)
            return pos, _finalize_generation_stats(stats, success=True, mean_distance=mean_distance)
        stats["layout_reject_incomplete"] = int(stats["layout_reject_incomplete"]) + 1
    raise RuntimeError(f"Failed to place set1 dots (n={n}).")

def _generate_set3_positions_tracked(
    n: int,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    *,
    layout_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    stats = _new_generation_stats()
    stats["target_dot_count"] = int(n)
    if n == 0:
        return [], _finalize_generation_stats(stats, success=True, mean_distance=_mean_interpoint_distance([]))
    if MplPath is None:
        raise RuntimeError("matplotlib is required for set3 (convex hull constraint).")
    cx, cy = cfg.W // 2, cfg.H // 2
    theta0 = rng.uniform(0.0, 2 * np.pi)
    verts = [
        (
            cx + cfg.set3_pentagon_radius * math.cos(theta0 + i * 2 * np.pi / 5),
            cy + cfg.set3_pentagon_radius * math.sin(theta0 + i * 2 * np.pi / 5),
        )
        for i in range(5)
    ]
    pent_path = _make_region_path(verts)
    radii = cfg.base_r + cfg.jitter * rng.standard_normal(n)
    radii = np.clip(radii, 0.1, None)
    for _ in range(int(layout_attempts)):
        stats["layout_attempts"] = int(stats["layout_attempts"]) + 1
        pos, placement_stats = _place_nonoverlapping(
            cfg.H,
            cfg.W,
            radii,
            rng,
            region_path=pent_path,
            distance_factor=cfg.set3_distance_factor,
            max_attempts_total=int(placement_max_attempts_total),
            per_item_attempts=int(placement_per_item_attempts),
            min_distance=cfg.min_dist,
            border_margin=cfg.border_margin,
            return_stats=True,
        )
        _merge_placement_stats(stats, placement_stats)
        if len(pos) == n:
            mean_distance = _mean_interpoint_distance(pos)
            return pos, _finalize_generation_stats(stats, success=True, mean_distance=mean_distance)
        stats["layout_reject_incomplete"] = int(stats["layout_reject_incomplete"]) + 1
    raise RuntimeError(f"Failed to place set3 dots (n={n}).")

def _try_set2_layouts(
    radii: np.ndarray,
    n: int,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    stats: Dict[str, object],
    *,
    set2_distance_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
    enforce_distance_match: bool,
) -> Optional[Tuple[List[Tuple[int, int, float]], float]]:
    for _ in range(int(set2_distance_attempts)):
        stats["layout_attempts"] = int(stats["layout_attempts"]) + 1
        pos, placement_stats = _place_nonoverlapping(
            cfg.H,
            cfg.W,
            radii,
            rng,
            max_attempts_total=int(placement_max_attempts_total),
            per_item_attempts=int(placement_per_item_attempts),
            min_distance=cfg.min_dist,
            border_margin=cfg.border_margin,
            return_stats=True,
        )
        _merge_placement_stats(stats, placement_stats)
        if len(pos) != n:
            stats["layout_reject_incomplete"] = int(stats["layout_reject_incomplete"]) + 1
            continue
        average_dist = _mean_interpoint_distance(pos)
        if not enforce_distance_match:
            return pos, average_dist
        if cfg.set2_target_dist_range[0] <= average_dist <= cfg.set2_target_dist_range[1]:
            return pos, average_dist
        stats["layout_reject_distance"] = int(stats["layout_reject_distance"]) + 1
    return None

def _generate_set2_positions_uniform_area_split(
    colors: np.ndarray,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    *,
    min_area_per_dot: float,
    set2_area_attempts: int,
    set2_distance_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
    enforce_distance_match: bool,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    stats = _new_generation_stats()
    n = int(colors.shape[0])
    stats["target_dot_count"] = n
    if n == 0:
        return [], _finalize_generation_stats(stats, success=True, mean_distance=_mean_interpoint_distance([]))

    white_mask = colors.astype(bool)
    black_mask = ~white_mask
    white_target_area, black_target_area = _target_areas_for_color_split(colors, cfg)

    for _ in range(int(set2_area_attempts)):
        stats["radius_sample_attempts"] = int(stats["radius_sample_attempts"]) + 1
        radii = np.zeros(n, dtype=np.float32)
        for mask, target_area in ((white_mask, white_target_area), (black_mask, black_target_area)):
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            radii[mask] = _sample_radii_from_uniform_area_budget(
                count,
                float(target_area),
                rng,
                float(min_area_per_dot),
            )

        placed = _try_set2_layouts(
            radii,
            n,
            rng,
            cfg,
            stats,
            set2_distance_attempts=int(set2_distance_attempts),
            placement_max_attempts_total=int(placement_max_attempts_total),
            placement_per_item_attempts=int(placement_per_item_attempts),
            enforce_distance_match=bool(enforce_distance_match),
        )
        if placed is not None:
            pos, average_dist = placed
            return pos, _finalize_generation_stats(stats, success=True, mean_distance=average_dist)
    n_white = int(np.count_nonzero(white_mask))
    n_black = int(np.count_nonzero(black_mask))
    raise RuntimeError(f"Failed to place set2 dots with random-uniform areas (n={n}, w={n_white}, b={n_black}).")

def _generate_set2_positions_uniform_area_global(
    n: int,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    *,
    min_area_per_dot: float,
    set2_area_attempts: int,
    set2_distance_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
    enforce_distance_match: bool,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    stats = _new_generation_stats()
    stats["target_dot_count"] = int(n)
    if n == 0:
        return [], _finalize_generation_stats(stats, success=True, mean_distance=_mean_interpoint_distance([]))

    for _ in range(int(set2_area_attempts)):
        stats["radius_sample_attempts"] = int(stats["radius_sample_attempts"]) + 1
        radii = _sample_radii_from_uniform_area_budget(n, float(cfg.set2_total_area), rng, float(min_area_per_dot))
        placed = _try_set2_layouts(
            radii,
            n,
            rng,
            cfg,
            stats,
            set2_distance_attempts=int(set2_distance_attempts),
            placement_max_attempts_total=int(placement_max_attempts_total),
            placement_per_item_attempts=int(placement_per_item_attempts),
            enforce_distance_match=bool(enforce_distance_match),
        )
        if placed is not None:
            pos, average_dist = placed
            return pos, _finalize_generation_stats(stats, success=True, mean_distance=average_dist)
    raise RuntimeError(f"Failed to place set2 dots with random-uniform global area (n={n}).")

def _make_set2_positions(
    total: int,
    colors: np.ndarray,
    rng: np.random.Generator,
    cfg: StimuliConfig,
    *,
    min_area_per_dot: float,
    set2_area_attempts: int,
    set2_distance_attempts: int,
    placement_max_attempts_total: int,
    placement_per_item_attempts: int,
    enforce_distance_match: bool,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, object]]:
    if cfg.set2_split_total_area_by_color:
        return _generate_set2_positions_uniform_area_split(
            colors,
            rng,
            cfg,
            min_area_per_dot=min_area_per_dot,
            set2_area_attempts=int(set2_area_attempts),
            set2_distance_attempts=int(set2_distance_attempts),
            placement_max_attempts_total=int(placement_max_attempts_total),
            placement_per_item_attempts=int(placement_per_item_attempts),
            enforce_distance_match=bool(enforce_distance_match),
        )
    return _generate_set2_positions_uniform_area_global(
        total,
        rng,
        cfg,
        min_area_per_dot=min_area_per_dot,
        set2_area_attempts=int(set2_area_attempts),
        set2_distance_attempts=int(set2_distance_attempts),
        placement_max_attempts_total=int(placement_max_attempts_total),
        placement_per_item_attempts=int(placement_per_item_attempts),
        enforce_distance_match=bool(enforce_distance_match),
    )
