"""Subspace and manifold alignment metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def orthonormal_basis(samples: Array, rank: int | None = None) -> Array:
    """Return right singular vectors spanning the dominant feature subspace."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
