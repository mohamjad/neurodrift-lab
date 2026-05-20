"""Trajectory-level drift metrics for latent intent dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryDrift:
    """Movement summary for two latent trajectories."""

    mean_position_shift: float
    mean_velocity_shift: float
    path_length_ratio: float
    curvature_shift: float


def path_length(trajectory: Array) -> float:
    """Return total Euclidean path length for a trajectory shaped ``(time, dims)``."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be shaped (time, dims)")
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())


def discrete_curvature(trajectory: Array, epsilon: float = 1e-8) -> Array:
    """Estimate pointwise curvature from finite differences."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[0] < 3:
        return np.zeros(max(trajectory.shape[0] - 2, 0), dtype=np.float64)
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)
    speed = np.maximum(np.linalg.norm(velocity[:-1], axis=1), epsilon)
    accel_norm = np.linalg.norm(acceleration, axis=1)
    return accel_norm / (speed**2)


def trajectory_drift(source: Array, target: Array) -> TrajectoryDrift:
    """Compare two aligned latent trajectories."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source and target trajectories must have the same shape")
    source_velocity = np.diff(source, axis=0)
    target_velocity = np.diff(target, axis=0)
    source_curvature = discrete_curvature(source)
    target_curvature = discrete_curvature(target)
    return TrajectoryDrift(
        mean_position_shift=float(np.linalg.norm(source - target, axis=1).mean()),
        mean_velocity_shift=float(np.linalg.norm(source_velocity - target_velocity, axis=1).mean()),
        path_length_ratio=path_length(target) / max(path_length(source), 1e-8),
        curvature_shift=float(np.abs(source_curvature - target_curvature).mean())
        if source_curvature.size
        else 0.0,
    )
