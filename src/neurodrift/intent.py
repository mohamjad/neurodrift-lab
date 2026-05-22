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


def intent_distribution_from_vectors(
    vectors: Array,
    labels: tuple[str, ...] | None = None,
    temperature: float = 1.0,
) -> IntentDistribution:
    """Convert continuous intent vectors into a weak aggregate distribution."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("vectors must be shaped (trials, intent_dims)")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("vectors must be non-empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if labels is None:
        signed_labels = tuple(
            label
            for idx in range(vectors.shape[1])
            for label in (f"intent_{idx}_pos", f"intent_{idx}_neg")
        )
        evidence_by_label = []
        for idx in range(vectors.shape[1]):
            evidence_by_label.append(np.maximum(vectors[:, idx], 0.0))
            evidence_by_label.append(np.maximum(-vectors[:, idx], 0.0))
        evidence = np.column_stack(evidence_by_label).mean(axis=0)
        evidence = (evidence - evidence.mean()) / max(float(evidence.std()), 1e-8)
        scores = dict(zip(signed_labels, evidence, strict=True))
        return soft_intent_from_observations(signed_labels, scores, temperature=temperature)

    if len(labels) != vectors.shape[1]:
        raise ValueError("labels must match intent dimension")

    evidence = np.mean(np.abs(vectors), axis=0)
    evidence = (evidence - evidence.mean()) / max(float(evidence.std()), 1e-8)
    scores = dict(zip(labels, evidence, strict=True))
    return soft_intent_from_observations(labels, scores, temperature=temperature)


def trial_intent_probabilities(vectors: Array, temperature: float = 1.0) -> Array:
    """Return per-trial weak intent probabilities from continuous labels."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("vectors must be shaped (trials, intent_dims)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    evidence = np.abs(vectors)
    evidence = evidence / np.maximum(evidence.std(axis=0, keepdims=True), 1e-8)
    shifted = (evidence - evidence.max(axis=1, keepdims=True)) / temperature
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def mean_distribution_entropy(probabilities: Array) -> float:
    """Mean base-2 entropy for a matrix of weak intent probabilities."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be shaped (trials, labels)")
    safe = np.maximum(probabilities, 1e-12)
    entropy = -(safe * np.log2(safe)).sum(axis=1)
    return float(entropy.mean())


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
