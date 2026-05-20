"""Metrics for intent preservation at the decoder output layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DecoderDrift:
    """Decoder-output stability summary."""
