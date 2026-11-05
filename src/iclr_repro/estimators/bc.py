"""Preserved Gaussian Bhattacharyya coefficient, without ridge repair."""
from __future__ import annotations
import numpy as np


def bhattacharyya_all_pairs(
    means: np.ndarray,
    covariances: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int = 32768,
) -> dict[str, np.ndarray]:
    """Compute Gaussian Bhattacharyya terms for arbitrary cell pairs."""

    mu = np.asarray(means, dtype=np.float64)
    cov = np.asarray(covariances, dtype=np.float64)
    first = np.asarray(source, dtype=np.int64).reshape(-1)
    second = np.asarray(target, dtype=np.int64).reshape(-1)
    if first.shape != second.shape or not len(first):
        raise ValueError("Invalid pair arrays")
    signs, cell_logdet = np.linalg.slogdet(cov)
    if np.any(signs <= 0):
        raise RuntimeError("Cell covariance is not positive definite")

    total = np.empty(len(first), dtype=np.float64)
    mean_term = np.empty(len(first), dtype=np.float64)
    covariance_term = np.empty(len(first), dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, len(first), step):
        stop = min(start + step, len(first))
        i = first[start:stop]
        j = second[start:stop]
        delta = mu[i] - mu[j]
        sigma_bar = 0.5 * (cov[i] + cov[j])
        solved = np.linalg.solve(sigma_bar, delta[..., None])[..., 0]
        current_mean = 0.125 * np.einsum("bi,bi->b", delta, solved, optimize=True)
        sign_bar, logdet_bar = np.linalg.slogdet(sigma_bar)
        if np.any(sign_bar <= 0):
            raise RuntimeError("A pooled pair covariance is not positive definite")
        current_covariance = 0.5 * (
            logdet_bar - 0.5 * (cell_logdet[i] + cell_logdet[j])
        )
        current_total = current_mean + current_covariance
        if float(np.min(current_total)) < -1.0e-8:
            raise RuntimeError(f"Negative Bhattacharyya distance: {float(np.min(current_total))}")
        total[start:stop] = np.maximum(current_total, 0.0)
        mean_term[start:stop] = current_mean
        covariance_term[start:stop] = current_covariance
    return {
        "distance": total,
        "mean_term": mean_term,
        "covariance_term": covariance_term,
        "overlap": np.exp(-total),
    }
