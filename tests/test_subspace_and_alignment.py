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


def test_procrustes_alignment_reduces_paired_trial_error() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(80, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    target = source @ q + 0.05 * rng.normal(size=(80, 8))

    raw_error = np.linalg.norm(source - target) / np.linalg.norm(source)
    aligned = ProcrustesAligner().fit(source, target).transform(target)
    aligned_error = np.linalg.norm(source - aligned) / np.linalg.norm(source)

    assert aligned_error < raw_error
    assert procrustes_error(target, source) < raw_error

