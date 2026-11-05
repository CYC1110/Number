"""Preserved count-weighted PCA and exact cell-by-set balancing primitives."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence


EPS = 1e-12

FOLD_NAMES = ("A", "B", "C")

HALF_NAMES = ("H1", "H2")

def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)

def build_cell_registry(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"N", "w", "b", "w_ratio", "set_id", "path"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Stimulus table is missing columns: {sorted(missing)}")
    work = table.copy().reset_index(drop=True)
    work["row_id"] = np.arange(len(work), dtype=np.int64)
    cells = (
        work[["N", "w", "b", "w_ratio"]]
        .drop_duplicates()
        .sort_values(["N", "w", "b"])
        .reset_index(drop=True)
    )
    cells["cell_id"] = np.arange(len(cells), dtype=np.int64)
    work = work.merge(cells, on=["N", "w", "b", "w_ratio"], how="left", validate="many_to_one")
    work = work.sort_values("row_id").reset_index(drop=True)
    if work["cell_id"].isna().any():
        raise RuntimeError("Failed to assign a cell_id to every stimulus")
    work["cell_id"] = work["cell_id"].astype(np.int64)
    cells = cells[["cell_id", "N", "w", "b", "w_ratio"]]
    return work, cells

def build_balanced_split_manifest(
    table: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create exact cell-by-set balanced 3-fold and split-half assignments."""
    if set(table["set_id"].unique()) != {1, 2, 3}:
        raise RuntimeError("Expected exactly set_id 1, 2, and 3")
    fold3 = np.full(len(table), -1, dtype=np.int8)
    half2 = np.full(len(table), -1, dtype=np.int8)
    audit_rows: list[dict[str, object]] = []

    for cell in cells.itertuples(index=False):
        by_set: dict[int, np.ndarray] = {}
        for set_id in (1, 2, 3):
            idx = table.index[
                (table["cell_id"] == int(cell.cell_id)) & (table["set_id"] == set_id)
            ].to_numpy(dtype=np.int64)
            if idx.size == 0:
                raise RuntimeError(f"Missing cell {(cell.w, cell.b)} in set {set_id}")
            by_set[set_id] = idx

        m3 = min(len(idx) // 3 for idx in by_set.values())
        m2 = min(len(idx) // 2 for idx in by_set.values())
        if m3 < 2 or m2 < 2:
            raise RuntimeError(
                f"Insufficient balanced repeats at cell {(cell.w, cell.b)}: m3={m3}, m2={m2}"
            )

        for set_id, idx in by_set.items():
            rng3 = np.random.default_rng(stable_seed(seed, "fold3", cell.w, cell.b, set_id))
            perm3 = rng3.permutation(idx)
            selected3 = perm3[: 3 * m3]
            for fold in range(3):
                fold3[selected3[fold * m3 : (fold + 1) * m3]] = fold

            rng2 = np.random.default_rng(stable_seed(seed, "half2", cell.w, cell.b, set_id))
            perm2 = rng2.permutation(idx)
            selected2 = perm2[: 2 * m2]
            for half in range(2):
                half2[selected2[half * m2 : (half + 1) * m2]] = half

            audit_rows.append(
                {
                    "cell_id": int(cell.cell_id),
                    "N": int(cell.N),
                    "w": int(cell.w),
                    "b": int(cell.b),
                    "set_id": int(set_id),
                    "n_available": int(len(idx)),
                    "n_per_fold3": int(m3),
                    "n_used_fold3": int(3 * m3),
                    "n_excluded_fold3": int(len(idx) - 3 * m3),
                    "n_per_half2": int(m2),
                    "n_used_half2": int(2 * m2),
                    "n_excluded_half2": int(len(idx) - 2 * m2),
                }
            )

    manifest = table[["row_id", "path", "set_id", "cell_id", "N", "w", "b", "w_ratio"]].copy()
    manifest["fold3_id"] = fold3
    manifest["fold3"] = pd.Categorical.from_codes(fold3, categories=list(FOLD_NAMES)).astype(object)
    manifest.loc[fold3 < 0, "fold3"] = "excluded"
    manifest["half2_id"] = half2
    manifest["half2"] = pd.Categorical.from_codes(half2, categories=list(HALF_NAMES)).astype(object)
    manifest.loc[half2 < 0, "half2"] = "excluded"

    # Exact balance is a hard invariant, not merely an audit statistic.
    for cell_id in cells["cell_id"].to_numpy(dtype=np.int64):
        for fold in range(3):
            counts = [
                int(((manifest["cell_id"] == cell_id) & (manifest["fold3_id"] == fold) & (manifest["set_id"] == s)).sum())
                for s in (1, 2, 3)
            ]
            if len(set(counts)) != 1:
                raise AssertionError(f"Unbalanced three-fold split: cell={cell_id}, fold={fold}, counts={counts}")
        for half in range(2):
            counts = [
                int(((manifest["cell_id"] == cell_id) & (manifest["half2_id"] == half) & (manifest["set_id"] == s)).sum())
                for s in (1, 2, 3)
            ]
            if len(set(counts)) != 1:
                raise AssertionError(f"Unbalanced split-half: cell={cell_id}, half={half}, counts={counts}")
    return manifest, pd.DataFrame(audit_rows)

def compute_centroids(
    activations: np.ndarray,
    table: pd.DataFrame,
    assignment: np.ndarray,
    *,
    n_groups: int,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(activations, dtype=np.float32)
    assignment = np.asarray(assignment, dtype=np.int64)
    dim = int(x.shape[1])
    means = np.full((n_groups, n_cells, dim), np.nan, dtype=np.float32)
    counts = np.zeros((n_groups, n_cells), dtype=np.int64)
    cell_ids = table["cell_id"].to_numpy(dtype=np.int64)
    for group in range(n_groups):
        for cell_id in range(n_cells):
            idx = np.flatnonzero((assignment == group) & (cell_ids == cell_id))
            if idx.size == 0:
                raise RuntimeError(f"Missing centroid: group={group}, cell={cell_id}")
            means[group, cell_id] = x[idx].mean(axis=0, dtype=np.float64).astype(np.float32)
            counts[group, cell_id] = int(idx.size)
    return means, counts

@dataclass(frozen=True)
class ConditionedBasis:
    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray

def _conditioned_basis_from_cell_centroids(
    cell_centroids: np.ndarray,
    cell_counts: np.ndarray,
    cells: pd.DataFrame,
    *,
    target: str,
    p_bins: int = 6,
) -> ConditionedBasis:
    centroids = np.asarray(cell_centroids, dtype=np.float32)
    counts = np.asarray(cell_counts, dtype=np.float64)
    if target == "N":
        level_ids = cells["N"].to_numpy(dtype=np.int64) - 1
        n_levels = 30
    elif target == "P":
        values = cells["w_ratio"].to_numpy(dtype=np.float64)
        # Match the established conditioned-PCA binning exactly: construct the
        # interior edges on [min, max], then expand only the two outer edges.
        # Expanding both endpoints before linspace would subtly move the 2/3 and
        # 5/6 boundaries and change their bin membership.
        edges = np.linspace(float(values.min()), float(values.max()), int(p_bins) + 1)
        edges[0] -= EPS
        edges[-1] += EPS
        level_ids = np.digitize(values, edges[1:-1], right=False)
        level_ids = np.clip(level_ids, 0, int(p_bins) - 1)
        n_levels = int(p_bins)
    else:
        raise ValueError(target)
    means = np.empty((n_levels, centroids.shape[1]), dtype=np.float32)
    for level in range(n_levels):
        mask = level_ids == level
        if not np.any(mask):
            raise RuntimeError(f"Missing conditioned-PCA level {target}={level}")
        level_weights = counts[mask]
        means[level] = np.average(centroids[mask], axis=0, weights=level_weights).astype(np.float32)
    effects = means - means.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(effects, full_matrices=False)
    energy = np.square(singular_values, dtype=np.float64)
    cumulative = np.cumsum(energy) / max(float(energy.sum()), EPS)
    return ConditionedBasis(
        basis=vh.T.astype(np.float32),
        singular_values=singular_values.astype(np.float64),
        cumulative_energy=cumulative.astype(np.float64),
    )

def conditioned_bases(
    cell_centroids: np.ndarray,
    cell_counts: np.ndarray,
    cells: pd.DataFrame,
) -> dict[str, ConditionedBasis]:
    return {
        "N": _conditioned_basis_from_cell_centroids(
            cell_centroids, cell_counts, cells, target="N"
        ),
        "P": _conditioned_basis_from_cell_centroids(
            cell_centroids, cell_counts, cells, target="P"
        ),
    }

def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray, rank_a: int, rank_b: int) -> np.ndarray:
    qa, _ = np.linalg.qr(np.asarray(basis_a[:, :rank_a], dtype=np.float64), mode="reduced")
    qb, _ = np.linalg.qr(np.asarray(basis_b[:, :rank_b], dtype=np.float64), mode="reduced")
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, 0.0, 1.0)))
