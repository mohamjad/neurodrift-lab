"""Synthetic BCI sessions with controllable neural manifold drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a paired source/target BCI simulation."""

    seed: int = 7
    trials: int = 96
    time_steps: int = 40
    channels: int = 16
    intent_dims: int = 2
    latent_dims: int = 6
    drift_strength: float = 0.15
    noise_scale: float = 0.08
    sample_rate_hz: float = 100.0


def _random_orthogonal(rng: np.random.Generator, dims: int) -> Array:
    matrix = rng.normal(size=(dims, dims))
    q, r = np.linalg.qr(matrix)
    return q @ np.diag(np.sign(np.diag(r)))


def _smooth_intents(rng: np.random.Generator, trials: int, intent_dims: int) -> Array:
    raw = rng.normal(size=(trials, intent_dims))
    kernel = np.array([0.15, 0.25, 0.2, 0.25, 0.15])
    padded = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.vstack(
        [(padded[idx : idx + kernel.size] * kernel[:, None]).sum(axis=0) for idx in range(trials)]
    )
    scale = np.maximum(smoothed.std(axis=0, keepdims=True), 1e-8)
    return (smoothed - smoothed.mean(axis=0, keepdims=True)) / scale


def _temporal_basis(time_steps: int, latent_dims: int) -> Array:
    time = np.linspace(0.0, 1.0, time_steps)
    basis = []
    for freq in range(1, latent_dims + 1):
        wave = np.sin(2 * np.pi * freq * time) if freq % 2 else np.cos(np.pi * freq * time)
        basis.append(wave)
    return np.asarray(basis, dtype=np.float64)


def simulate_session_pair(config: SimulationConfig) -> SessionPair:
    """Create source and target sessions with known latent and covariance drift."""

    if config.intent_dims > config.latent_dims:
        raise ValueError("intent_dims must be less than or equal to latent_dims")
    rng = np.random.default_rng(config.seed)
    intents = _smooth_intents(rng, config.trials, config.intent_dims)
    intent_to_latent = rng.normal(scale=0.8, size=(config.intent_dims, config.latent_dims))
    latent_static = intents @ intent_to_latent
    temporal = _temporal_basis(config.time_steps, config.latent_dims)
    latent = latent_static[:, None, :] + 0.25 * temporal.T[None, :, :]

    source_projection = rng.normal(scale=0.6, size=(config.latent_dims, config.channels))
    rotation = _random_orthogonal(rng, config.channels)
    gain = np.linspace(1.0, 1.0 + config.drift_strength, config.channels)
    target_projection = source_projection @ (
        (1.0 - config.drift_strength) * np.eye(config.channels)
        + config.drift_strength * rotation
    )
    target_projection = target_projection * gain[None, :]
