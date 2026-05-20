"""Subspace and manifold alignment metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def orthonormal_basis(samples: Array, rank: int | None = None) -> Array:
    """Return right singular vectors spanning the dominant feature subspace."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be shaped (observations, features)")
    centered = samples - samples.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if rank is None:
        rank = min(samples.shape)
    if rank <= 0:
        raise ValueError("rank must be positive")
    return vh[:rank].T


def principal_angles(source_basis: Array, target_basis: Array) -> Array:
    """Return principal angles in radians between two column-orthonormal bases."""

    source_basis = np.asarray(source_basis, dtype=np.float64)
    target_basis = np.asarray(target_basis, dtype=np.float64)
