from __future__ import annotations

import numpy as np

from neurodrift.metrics.subspace import procrustes_error, subspace_distance
from neurodrift.models.alignment import ProcrustesAligner, WhiteningColoringAligner


def test_subspace_distance_is_near_zero_for_rotated_basis() -> None:
    rng = np.random.default_rng(3)
