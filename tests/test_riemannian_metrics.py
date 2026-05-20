from __future__ import annotations

import numpy as np

from neurodrift.metrics.riemannian import affine_invariant_distance, covariance_drift
from neurodrift.numerics import safe_covariance


def test_affine_invariant_distance_is_zero_for_same_matrix() -> None:
    samples = np.random.default_rng(1).normal(size=(64, 5))
