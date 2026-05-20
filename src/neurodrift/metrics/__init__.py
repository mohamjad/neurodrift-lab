"""Metric functions for drift, geometry, and decoder stability."""

from neurodrift.metrics.composite import DriftReport, build_drift_report
from neurodrift.metrics.riemannian import (
    affine_invariant_distance,
    covariance_drift,
    log_euclidean_distance,
)
from neurodrift.metrics.subspace import principal_angles, subspace_distance
from neurodrift.metrics.trajectory import trajectory_drift

__all__ = [
    "DriftReport",
    "affine_invariant_distance",
    "build_drift_report",
    "covariance_drift",
    "log_euclidean_distance",
    "principal_angles",
    "subspace_distance",
    "trajectory_drift",
]
