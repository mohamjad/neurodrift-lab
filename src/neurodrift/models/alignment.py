"""Alignment transforms for adapting target sessions to a source decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from neurodrift.metrics.subspace import orthogonal_procrustes
from neurodrift.numerics import matrix_invsqrt_spd, matrix_sqrt_spd, safe_covariance

Array = NDArray[np.float64]


class Aligner(Protocol):
    """Protocol for session alignment transforms."""

    def fit(self, source: Array, target: Array) -> Aligner:
        """Fit the transform from target-space features into source-space features."""

    def transform(self, target: Array) -> Array:
        """Map target-space features into source-space features."""


@dataclass
class IdentityAligner:
    """No-op alignment baseline."""

    def fit(self, source: Array, target: Array) -> IdentityAligner:
        return self

    def transform(self, target: Array) -> Array:
        return np.asarray(target, dtype=np.float64)


@dataclass
class CenteringTransform:
    """Mean-shift target features to match the source session mean."""

    source_mean_: Array | None = None
    target_mean_: Array | None = None

    def fit(self, source: Array, target: Array) -> CenteringTransform:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.source_mean_ = source.mean(axis=0, keepdims=True)
        self.target_mean_ = target.mean(axis=0, keepdims=True)
        return self

    def transform(self, target: Array) -> Array:
        if self.source_mean_ is None or self.target_mean_ is None:
            raise RuntimeError("aligner must be fit before transform")
        return np.asarray(target, dtype=np.float64) - self.target_mean_ + self.source_mean_


@dataclass
class ProcrustesAligner:
    """Orthogonal alignment based on paired trial features."""
