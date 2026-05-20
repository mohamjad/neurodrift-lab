"""Decoder baselines used by evaluation environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RidgeDecoder:
    """Small deterministic ridge decoder for neural features to intent vectors."""

    alpha: float = 1e-3
    weights_: Array | None = None
    bias_: Array | None = None

    def fit(self, features: Array, intent: Array) -> RidgeDecoder:
        """Fit decoder weights."""

        features = np.asarray(features, dtype=np.float64)
        intent = np.asarray(intent, dtype=np.float64)
