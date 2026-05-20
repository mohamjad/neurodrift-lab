"""Riemannian metrics for covariance movement across sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.numerics import (
    frobenius_norm,
    matrix_invsqrt_spd,
    matrix_log_spd,
    regularize_spd,
    safe_covariance,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CovarianceDrift:
    """Summary of SPD covariance movement between two sessions."""

    source_covariance: Array
    target_covariance: Array
    affine_invariant: float
    log_euclidean: float
    covariance_trace_ratio: float


def affine_invariant_distance(source: Array, target: Array, epsilon: float = 1e-8) -> float:
    """Affine-invariant geodesic distance on SPD matrices.

    The distance is ``||log(A^{-1/2} B A^{-1/2})||_F``. It is a standard
