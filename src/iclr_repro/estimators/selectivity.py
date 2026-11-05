"""Preserved vectorized two-way ANOVA; use balanced single-color stimuli."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import stats


@dataclass
class SelectivityConfig:
    alpha: float = 0.01
    interaction_alpha: float = 0.01
    legacy_criteria: bool = True
    max_units: Optional[int] = None
    batch_size: int = 128
    n_jobs: int = -1

def compute_selectivity(
    conv5_flat: np.ndarray,
    numeros: np.ndarray,
    set_id: np.ndarray,
    cfg: SelectivityConfig,
) -> Dict[str, np.ndarray]:
    """Vectorised two-way ANOVA for number selectivity."""
    conv = np.asarray(conv5_flat, dtype=np.float64)
    numeros = np.asarray(numeros)
    set_id = np.asarray(set_id)

    if conv.ndim != 2:
        raise ValueError("conv5_flat must be a 2D array (samples × units).")
    if numeros.shape[0] != conv.shape[0] or set_id.shape[0] != conv.shape[0]:
        raise ValueError("numeros and set_id must align with conv5_flat rows.")

    N, F_total = conv.shape
    F_eval = F_total if cfg.max_units is None else min(F_total, int(cfg.max_units))

    p_num = np.full(F_total, np.nan, dtype=np.float32)
    p_set = np.full(F_total, np.nan, dtype=np.float32)
    p_int = np.full(F_total, np.nan, dtype=np.float32)
    f_num = np.full(F_total, np.nan, dtype=np.float32)
    eta2 = np.full(F_total, np.nan, dtype=np.float32)
    preferred = np.full(F_total, -1, dtype=np.int32)
    preferred_by_set = np.full((0, F_total), -1, dtype=np.int32)

    if F_eval <= 0 or N == 0:
        return dict(
            is_selective=np.zeros(F_total, dtype=bool),
            p_num=p_num,
            p_set=p_set,
            p_inter=p_int,
            f_num=f_num,
            eta2=eta2,
            preferred=preferred,
        )

    conv_eval = conv[:, :F_eval]
    if not np.isfinite(conv_eval).all():
        conv_eval = np.where(np.isfinite(conv_eval), conv_eval, 0.0)

    num_levels, num_idx = np.unique(numeros, return_inverse=True)
    set_levels, set_idx = np.unique(set_id, return_inverse=True)
    A = int(num_levels.size)
    B = int(set_levels.size)

    if A <= 1 or B <= 1:
        return dict(
            is_selective=np.zeros(F_total, dtype=bool),
            p_num=p_num,
            p_set=p_set,
            p_inter=p_int,
            f_num=f_num,
            eta2=eta2,
            preferred=preferred,
        )

    preferred_by_set = np.full((B, F_total), -1, dtype=np.int32)

    group_idx = num_idx * B + set_idx
    G = A * B

    counts = np.bincount(group_idx, minlength=G).astype(np.float64)

    sums = np.zeros((G, F_eval), dtype=np.float64)
    np.add.at(sums, group_idx, conv_eval)
    sums_sq = np.zeros((G, F_eval), dtype=np.float64)
    np.add.at(sums_sq, group_idx, conv_eval ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        ss_within = sums_sq - (sums ** 2) / counts[:, None]
    ss_within[counts == 0, :] = 0.0
    SSE = ss_within.sum(axis=0)

    grand_mean = conv_eval.mean(axis=0)
    SS_total = ((conv_eval - grand_mean) ** 2).sum(axis=0)

    sums_reshaped = sums.reshape(A, B, F_eval)
    counts_reshaped = counts.reshape(A, B)

    sums_a = sums_reshaped.sum(axis=1)
    counts_a = counts_reshaped.sum(axis=1)
    mean_a = np.divide(
        sums_a,
        counts_a[:, None],
        out=np.zeros_like(sums_a),
        where=counts_a[:, None] > 0,
    )

    sums_b = sums_reshaped.sum(axis=0)
    counts_b = counts_reshaped.sum(axis=0)
    mean_b = np.divide(
        sums_b,
        counts_b[:, None],
        out=np.zeros_like(sums_b),
        where=counts_b[:, None] > 0,
    )

    SSA = ((mean_a - grand_mean) ** 2 * counts_a[:, None]).sum(axis=0)
    SSB = ((mean_b - grand_mean) ** 2 * counts_b[:, None]).sum(axis=0)

    means_ij = np.divide(
        sums_reshaped,
        counts_reshaped[:, :, None],
        out=np.zeros_like(sums_reshaped),
        where=counts_reshaped[:, :, None] > 0,
    )
    interaction = (
        means_ij
        - mean_a[:, None, :]
        - mean_b[None, :, :]
        + grand_mean[None, None, :]
    )
    SSAB = (interaction**2 * counts_reshaped[:, :, None]).sum(axis=(0, 1))

    df_a = A - 1
    df_b = B - 1
    df_ab = df_a * df_b
    df_error = N - G

    with np.errstate(invalid="ignore", divide="ignore"):
        MSE = SSE / df_error if df_error > 0 else np.full(F_eval, np.nan, dtype=np.float64)
        MS_num = SSA / df_a if df_a > 0 else np.full(F_eval, np.nan, dtype=np.float64)
        MS_set = SSB / df_b if df_b > 0 else np.full(F_eval, np.nan, dtype=np.float64)
        MS_int = SSAB / df_ab if df_ab > 0 else np.full(F_eval, np.nan, dtype=np.float64)

    F_num_eval = np.divide(
        MS_num,
        MSE,
        out=np.full_like(MS_num, np.nan),
        where=MSE > 0,
    )
    F_set_eval = np.divide(
        MS_set,
        MSE,
        out=np.full_like(MS_set, np.nan),
        where=MSE > 0,
    )
    F_int_eval = np.divide(
        MS_int,
        MSE,
        out=np.full_like(MS_int, np.nan),
        where=MSE > 0,
    )

    p_num_eval = stats.f.sf(F_num_eval, df_a, df_error) if df_a > 0 and df_error > 0 else np.full(F_eval, np.nan)
    p_set_eval = stats.f.sf(F_set_eval, df_b, df_error) if df_b > 0 and df_error > 0 else np.full(F_eval, np.nan)
    p_int_eval = stats.f.sf(F_int_eval, df_ab, df_error) if df_ab > 0 and df_error > 0 else np.full(F_eval, np.nan)

    eta_eval = np.divide(
        SSA,
        SSA + SSE,
        out=np.zeros_like(SSA),
        where=(SSA + SSE) > 0,
    )

    pref_idx = np.argmax(mean_a, axis=0)
    preferred_eval = num_levels[pref_idx]
    per_set_idx = np.argmax(means_ij, axis=0)
    preferred_sets_eval = num_levels[per_set_idx]

    p_num[:F_eval] = p_num_eval.astype(np.float32)
    p_set[:F_eval] = p_set_eval.astype(np.float32)
    p_int[:F_eval] = p_int_eval.astype(np.float32)
    f_num[:F_eval] = F_num_eval.astype(np.float32)
    eta2[:F_eval] = eta_eval.astype(np.float32)
    preferred[:F_eval] = preferred_eval.astype(np.int32)
    preferred_by_set[:, :F_eval] = preferred_sets_eval.astype(np.int32)

    is_selective = (p_num < cfg.alpha) & (p_set > cfg.alpha) & (p_int > cfg.interaction_alpha)

    return dict(
        is_selective=is_selective,
        p_num=p_num,
        p_set=p_set,
        p_inter=p_int,
        f_num=f_num,
        eta2=eta2,
        preferred=preferred,
        preferred_by_set=preferred_by_set,
    )
