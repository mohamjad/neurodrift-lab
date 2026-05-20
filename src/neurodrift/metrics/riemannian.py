"""Riemannian metrics for covariance movement across sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.numerics import (
