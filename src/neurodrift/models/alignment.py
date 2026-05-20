"""Alignment transforms for adapting target sessions to a source decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from neurodrift.metrics.subspace import orthogonal_procrustes
from neurodrift.numerics import matrix_invsqrt_spd, matrix_sqrt_spd, safe_covariance

Array = NDArray[np.float64]

