"""Trajectory-level drift metrics for latent intent dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryDrift:
    """Movement summary for two latent trajectories."""
