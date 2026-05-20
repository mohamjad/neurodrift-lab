"""Shared data containers for longitudinal BCI sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SessionBatch:
    """Neural observations and aligned intent labels for one recording session.

    Attributes:
        neural: Array shaped ``(trials, time, channels)``.
        intent: Array shaped ``(trials, intent_dims)``.
        session_id: Stable label for provenance and reporting.
        sample_rate_hz: Sampling rate for temporal metrics.
    """

    neural: Array
    intent: Array
    session_id: str
    sample_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        if self.neural.ndim != 3:
            raise ValueError("neural must be shaped (trials, time, channels)")
        if self.intent.ndim != 2:
            raise ValueError("intent must be shaped (trials, intent_dims)")
        if self.neural.shape[0] != self.intent.shape[0]:
            raise ValueError("neural and intent must have the same trial count")
