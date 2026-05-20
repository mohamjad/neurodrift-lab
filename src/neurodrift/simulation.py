"""Synthetic BCI sessions with controllable neural manifold drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]


@dataclass(frozen=True)
