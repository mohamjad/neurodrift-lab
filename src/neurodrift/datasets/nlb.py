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

    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise ValueError(f"dataset group not found: {dataset}")
        group = handle[dataset]
        spikes = np.asarray(group[spikes_key], dtype=np.float64)
        behavior = np.asarray(group[behavior_key], dtype=np.float64)
    if spikes.ndim != 3:
        raise ValueError("NLB spikes must be shaped (trials, time, channels)")
    intent = _trial_intent(behavior)
    trial_count = min(spikes.shape[0], intent.shape[0])
    if max_trials is not None:
        trial_count = min(trial_count, max_trials)
    if trial_count < 4:
        raise ValueError("at least four trials are required to build an NLB pair")
    spikes = spikes[:trial_count]
    intent = intent[:trial_count]
    intent_scale = np.maximum(intent.std(axis=0, keepdims=True), 1e-8)
    intent = (intent - intent.mean(axis=0, keepdims=True)) / intent_scale
    source_idx = np.arange(0, trial_count, 2)
    target_idx = np.arange(1, trial_count, 2)
    paired_count = min(source_idx.size, target_idx.size)
    source_idx = source_idx[:paired_count]
    target_idx = target_idx[:paired_count]
    return SessionPair(
        source=SessionBatch(
            neural=spikes[source_idx],
            intent=intent[source_idx],
            session_id=f"{dataset}-even",
            sample_rate_hz=50.0,
        ),
        target=SessionBatch(
            neural=spikes[target_idx],
            intent=intent[target_idx],
