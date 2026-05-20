"""Neural Latents Benchmark HDF5 adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]


def _require_h5py():
