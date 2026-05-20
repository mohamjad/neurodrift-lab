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
    if behavior.ndim == 2:
        return behavior
    raise ValueError("NLB behavior must be shaped (trials, time, dims) or (trials, dims)")


def load_nlb_h5_pair(
    path: Path,
    *,
    dataset: str = "mc_maze_small_20",
    spikes_key: str = "eval_spikes_heldout",
    behavior_key: str = "eval_behavior",
    max_trials: int | None = 80,
) -> SessionPair:
    """Load one NLB dataset group as an even/odd source-target session pair.

    NLB is not longitudinal by itself. This adapter creates a real neural
    population smoke test by splitting held-out evaluation trials into two
    matched pseudo-sessions while preserving the original spike and behavior
    arrays.
    """
