"""Decoder baselines used by evaluation environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RidgeDecoder:
    """Small deterministic ridge decoder for neural features to intent vectors."""
