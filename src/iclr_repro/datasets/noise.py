"""Preserved fixed per-index Gaussian RGB noise source."""
from __future__ import annotations
import numpy as np


class NoiseSource:
    """Generate the historical 105k control's uint8 image for each sample ID."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, kind: str, resolution: int, seed: int, shader_manifest=None):
        if kind != "noise":
            raise ValueError(f"Expected noise condition, got {kind}")
        if resolution != 256 or shader_manifest not in (None, "None"):
            raise ValueError("Fixed-noise SimCLR uses 256x256 generated sources and no image manifest")
        self.resolution = resolution
        self.seed = seed

    def __call__(self, sample_id: int) -> np.ndarray:
        if sample_id < 0:
            raise ValueError("sample_id must be nonnegative")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(sample_id)]))
        gaussian = rng.standard_normal((self.resolution, self.resolution, 3), dtype=np.float32)
        pixels = np.rint(np.clip((gaussian * self.STD + self.MEAN) * 255.0, 0.0, 255.0))
        return np.ascontiguousarray(pixels.astype(np.uint8))
