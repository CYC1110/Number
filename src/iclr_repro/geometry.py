"""N1/P2 geometry: descriptive full-data angle and held-out all-pair BC."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .estimators.basis import (build_cell_registry, build_balanced_split_manifest,
                              conditioned_bases, ConditionedBasis, principal_angles)
from .estimators.bc import bhattacharyya_all_pairs

P_EDGES = np.arange(1, 12, 2, dtype=float) / 12


def p_classes(n, w):
    n, w = np.asarray(n), np.asarray(w)
    if np.any(n <= 0) or np.any(w < 0) or np.any(w > n):
        raise ValueError("P requires N > 0 and 0 <= W <= N")
    # Reconstruct fractions from integer counts; never parse rounded ratios.
    return np.digitize(w / n, P_EDGES)


def registry(metadata, seed=1729):
    table = metadata.copy().reset_index(drop=True)
    for key in ["N","w"]:
        values=table[key].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():
            raise ValueError("Dot counts must be finite integers")
    if (table.w<0).any() or (table.w>table.N).any():raise ValueError("Invalid white-dot count")
    table["w_ratio"] = table.w.to_numpy() / table.N.to_numpy()
    if "b" not in table:table["b"] = table.N - table.w
    if (table.N <= 0).any() or not (table.N == table.w + table.b).all():
        raise ValueError("Invalid nonblank dot counts")
    table, cells = build_cell_registry(table)
    split, balance = build_balanced_split_manifest(table, cells, seed=seed)
    table = table.merge(split[["row_id", "fold3_id"]], on="row_id", validate="one_to_one")
    return table, cells, balance


def cell_means(x, table, cells, indices):
    sums = np.zeros((len(cells), x.shape[1]), dtype=np.float64)
    ids = table.cell_id.to_numpy()[indices]
    np.add.at(sums, ids, x[indices])
    counts = np.bincount(ids, minlength=len(cells))
    if np.any(counts == 0):raise ValueError("Every cell needs training observations")
    return (sums / counts[:, None]).astype(np.float32), counts


def conditioned7(means, counts, cells):
    """Same count-weighted level means and unweighted centering as source P7."""
    n = conditioned_bases(means, counts, cells)["N"]
    labels = p_classes(cells.N.to_numpy(), cells.w.to_numpy())
    if set(labels) != set(range(7)):raise ValueError("All seven P classes are required")
    levels = np.array([np.average(means[labels == k], axis=0, weights=counts[labels == k])
                       for k in range(7)], dtype=np.float32)
    effects = levels - levels.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(effects, full_matrices=False)
    energy = np.square(singular, dtype=np.float64)
    if energy.sum() <= 1e-12 or np.square(n.singular_values).sum() <= 1e-12:
        raise ValueError("Quantity-conditioned subspaces are undefined")
    p = ConditionedBasis(vh.T.astype(np.float32), singular.astype(float),
                         np.cumsum(energy) / energy.sum())
    if len(p.singular_values) < 2 or p.singular_values[1] <= 1e-12:
        raise ValueError("P subspace has fewer than two identifiable directions")
    return {"N": n, "P": p}


def orthonormal_union(un, up):
    joined = np.column_stack([un[:, :1], up[:, :2]]).astype(np.float64)
    if np.linalg.matrix_rank(joined) != 3:
        raise ValueError("N1/P2 union is rank deficient; do not invent a 3D overlap")
    return np.linalg.qr(joined, mode="reduced")[0]


def gaussian_cells(z, ids, n_cells):
    means, covariances = [], []
    for cell in range(n_cells):
        values = z[ids == cell]
        if len(values) < 2:raise ValueError("Insufficient held-out observations")
        model = LedoitWolf(store_precision=False).fit(values)
        means.append(model.location_)
        covariances.append((model.covariance_ + model.covariance_.T) / 2)
    return np.asarray(means), np.asarray(covariances)


def measure(x, table, cells):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(table) or not np.isfinite(x).all():
        raise ValueError("Activations must be finite samples-by-features aligned to metadata")
    means, counts = cell_means(x, table, cells, np.arange(len(x)))
    full = conditioned7(means, counts, cells)
    un, up = full["N"].basis[:, :1], full["P"].basis[:, :2]
    angle = float(principal_angles(un, up, 1, 2)[0])
    i, j = np.triu_indices(len(cells), 1)
    folds, displays = [], []
    for held in range(3):
        assignments = table.fold3_id.to_numpy()
        train = np.flatnonzero((assignments >= 0) & (assignments != held))
        test = np.flatnonzero(assignments == held)
        m, c = cell_means(x, table, cells, train)
        basis = conditioned7(m, c, cells)
        q = orthonormal_union(basis["N"].basis, basis["P"].basis)
        z = x[test].astype(np.float64) @ q
        mu, cov = gaussian_cells(z, table.cell_id.to_numpy()[test], len(cells))
        positive = np.linalg.eigvalsh(cov).min(axis=1) > 0
        # A single undefined pair invalidates the canonical fold mean.
        if positive.all():
            bc = float(bhattacharyya_all_pairs(mu, cov, i, j)["overlap"].mean())
            status = "regular"
        else:
            bc, status = float("nan"), "undefined_singular_covariance"
        folds.append({"held_out_fold": held, "all_pair_bc": bc,
                      "n_pairs": len(i), "positive_covariance_cells": int(positive.sum()),
                      "n_cells": len(cells), "status": status})
        displays.append({"Q": q, "means": mu, "covariances": cov,
                         "U_N": basis["N"].basis[:, :1], "U_P": basis["P"].basis[:, :2]})
    result = {"principal_angle_deg": angle,
              "all_pair_bc": float(np.mean([f["all_pair_bc"] for f in folds])),
              "N_cumulative_ev": full["N"].cumulative_energy.tolist(),
              "P_cumulative_ev": full["P"].cumulative_energy.tolist(),
              "folds": folds, "n_images": len(x), "n_cells": len(cells),
              "angle_basis_scope": "full_data_descriptive",
              "bc_basis_scope": "other_two_folds", "covariance": "LedoitWolf",
              "normalization": "raw activations; angle/BC invariant to common nonzero scalar"}
    return result, {"U_N": un, "U_P": up}, displays


def ablate(x, basis):
    """Raw-origin projection removal; no centering and no decoder retraining."""
    u = np.asarray(basis)
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-5):
        raise ValueError("Ablation directions must be orthonormal")
    return np.asarray(x) - (np.asarray(x) @ u) @ u.T
