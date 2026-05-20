"""Neural Latents Benchmark HDF5 adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install neurodrift-lab[integrations] to load NLB HDF5 files") from exc
    return h5py


def _trial_intent(behavior: Array) -> Array:
    if behavior.ndim == 3:
        return behavior.mean(axis=1)
