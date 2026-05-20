from __future__ import annotations

import numpy as np

from neurodrift.metrics.subspace import procrustes_error, subspace_distance
from neurodrift.models.alignment import ProcrustesAligner, WhiteningColoringAligner


def test_subspace_distance_is_near_zero_for_rotated_basis() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(100, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    rotated = samples @ q

    assert subspace_distance(samples, rotated, rank=6) < 1e-7
