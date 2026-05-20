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
    if source_basis.ndim != 2 or target_basis.ndim != 2:
        raise ValueError("basis arrays must be matrices")
    if source_basis.shape[0] != target_basis.shape[0]:
        raise ValueError("bases must live in the same ambient dimension")
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    return np.arccos(np.clip(singular_values, -1.0, 1.0))


def subspace_distance(
    source_samples: Array,
    target_samples: Array,
    rank: int | None = None,
) -> float:
    """Return RMS principal-angle distance between dominant neural subspaces."""

    source_basis = orthonormal_basis(source_samples, rank)
    target_basis = orthonormal_basis(target_samples, rank)
    angles = principal_angles(source_basis, target_basis)
    return float(np.sqrt(np.mean(angles**2)))


def orthogonal_procrustes(source: Array, target: Array) -> Array:
    """Fit the orthogonal matrix that best maps source samples to target samples."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    cross_cov = source.T @ target
    u, _, vh = np.linalg.svd(cross_cov, full_matrices=False)
    return u @ vh


def procrustes_error(source: Array, target: Array) -> float:
    """Return normalized alignment error after orthogonal Procrustes."""
