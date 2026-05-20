"""Numerical helpers for stable small-to-medium neural geometry problems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def as_float_array(value: Array) -> Array:
    """Return a contiguous float64 array."""

    return np.asarray(value, dtype=np.float64)


def symmetrize(matrix: Array) -> Array:
    """Project a square matrix onto the symmetric matrices."""

    matrix = as_float_array(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return (matrix + matrix.T) / 2.0


def regularize_spd(matrix: Array, epsilon: float = 1e-6) -> Array:
    """Return a numerically SPD matrix by eigenvalue clipping."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    vals, vecs = np.linalg.eigh(symmetrize(matrix))
    vals = np.maximum(vals, epsilon)
    return symmetrize((vecs * vals) @ vecs.T)


def matrix_log_spd(matrix: Array, epsilon: float = 1e-8) -> Array:
    """Matrix logarithm for symmetric positive-definite matrices."""

    vals, vecs = np.linalg.eigh(regularize_spd(matrix, epsilon))
    return symmetrize((vecs * np.log(vals)) @ vecs.T)


def matrix_sqrt_spd(matrix: Array, epsilon: float = 1e-8) -> Array:
    """Matrix square root for symmetric positive-definite matrices."""

    vals, vecs = np.linalg.eigh(regularize_spd(matrix, epsilon))
    return symmetrize((vecs * np.sqrt(vals)) @ vecs.T)


def matrix_invsqrt_spd(matrix: Array, epsilon: float = 1e-8) -> Array:
