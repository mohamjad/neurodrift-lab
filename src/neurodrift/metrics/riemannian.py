"""Riemannian metrics for covariance movement across sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.numerics import (
    frobenius_norm,
    matrix_invsqrt_spd,
    matrix_log_spd,
    regularize_spd,
    safe_covariance,
