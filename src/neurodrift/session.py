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
