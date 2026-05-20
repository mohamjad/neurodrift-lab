"""Metrics for intent preservation at the decoder output layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DecoderDrift:
    """Decoder-output stability summary."""

    mse: float
    cosine_error: float
    gain_ratio: float
    bias_shift: float


def decoder_drift(source_outputs: Array, target_outputs: Array) -> DecoderDrift:
    """Compare paired decoder outputs from source and target sessions."""

