"""Numerical helpers for stable small-to-medium neural geometry problems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def as_float_array(value: Array) -> Array:
    """Return a contiguous float64 array."""

    return np.asarray(value, dtype=np.float64)

