"""Metric functions for drift, geometry, and decoder stability."""

from neurodrift.metrics.composite import DriftReport, build_drift_report
from neurodrift.metrics.riemannian import (
    affine_invariant_distance,
