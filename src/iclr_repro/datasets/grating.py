"""Preserved full-field sinusoidal grating source."""
from __future__ import annotations
import hashlib, math
import numpy as np


def sample_seed(dataset_seed: int, sample_id: int) -> int:
    return int(np.random.SeedSequence([dataset_seed, sample_id]).generate_state(1)[0])

def grating_parameters(sample_id: int, resolution: int, dataset_seed: int):
    r = np.random.default_rng(sample_seed(dataset_seed, sample_id))
    return dict(wavelength_px=float(r.uniform(3, 30) * resolution / 64),
                orientation_deg=float(r.uniform(0, 360)), phase_deg=float(r.uniform(0, 360)),
                center_x=float(r.uniform(.25, .75) * resolution),
                center_y=float(r.uniform(.25, .75) * resolution),
                sigma_px=float(r.uniform(.12, .30) * resolution))

def grating(sample_id: int, resolution: int = 256, dataset_seed: int = 20260916,
            localized: bool = False):
    """CUB create_grating sine equation, plus optional custom Gaussian envelope.

    sf in upstream code is wavelength in pixels, not cycles per pixel. Scale its
    [3,30] range at 64 px to preserve cycles/image. Full-field grating is inspired
    by CUB single-grating, not an exact reproduction of all CUB preprocessing.
    Localized=True is our specified Gabor variant, not the CUB stimulus.
    """
    p = grating_parameters(sample_id, resolution, dataset_seed)
    y, x = np.mgrid[:resolution, :resolution].astype(np.float32)
    angle = math.radians(p['orientation_deg'])
    carrier = np.sin(2 * math.pi * (math.sin(angle) * x - math.cos(angle) * y)
                     / p['wavelength_px'] + math.radians(p['phase_deg']))
    if localized:
        carrier *= np.exp(-((x-p['center_x'])**2 + (y-p['center_y'])**2)
                          / (2*p['sigma_px']**2))
    # Fixed mean background .5, carrier amplitude .45. Do not normalize away the
    # Gaussian envelope. Achieved RMS varies with envelope and is recorded in QA.
    a = np.rint(np.clip(.5 + .45 * carrier, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
