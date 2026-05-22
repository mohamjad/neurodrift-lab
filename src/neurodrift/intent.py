"""Weak intent supervision under neural change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class IntentDistribution:
    """Probabilistic intent label for one trial or utterance."""

    labels: tuple[str, ...]
    probabilities: Array

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 1:
            raise ValueError("probabilities must be a vector")
        if len(self.labels) != self.probabilities.shape[0]:
            raise ValueError("labels and probabilities must have the same length")
        if np.any(self.probabilities < 0):
            raise ValueError("probabilities cannot be negative")
        total = float(self.probabilities.sum())
        if not np.isclose(total, 1.0):
            raise ValueError("probabilities must sum to 1")

    @property
    def entropy(self) -> float:
        safe = np.maximum(self.probabilities, 1e-12)
        return float(-(safe * np.log2(safe)).sum())

    @property
    def confidence(self) -> float:
        return float(self.probabilities.max())


def soft_intent_from_observations(
    labels: tuple[str, ...],
    observations: dict[str, float],
    temperature: float = 1.0,
) -> IntentDistribution:
    """Build a weak intent label from noisy observation scores."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = np.array([observations.get(label, 0.0) for label in labels], dtype=np.float64)
    shifted = (raw - raw.max()) / temperature
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    return IntentDistribution(labels=labels, probabilities=probs)


def intent_distribution_distance(source: IntentDistribution, target: IntentDistribution) -> float:
    """Jensen-Shannon distance between two intent distributions."""

    if source.labels != target.labels:
        raise ValueError("intent distributions must share labels")
    midpoint = 0.5 * (source.probabilities + target.probabilities)
    return float(
        0.5 * _kl(source.probabilities, midpoint) + 0.5 * _kl(target.probabilities, midpoint)
    )


def _kl(left: Array, right: Array) -> float:
    left = np.maximum(left, 1e-12)
    right = np.maximum(right, 1e-12)
    return float(np.sum(left * np.log2(left / right)))
