"""Synthetic BCI sessions with controllable neural manifold drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a paired source/target BCI simulation."""

    seed: int = 7
    trials: int = 96
    time_steps: int = 40
    channels: int = 16
    intent_dims: int = 2
    latent_dims: int = 6
    drift_strength: float = 0.15
