"""Decoder baselines used by evaluation environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class RidgeDecoder:
    """Small deterministic ridge decoder for neural features to intent vectors."""

    alpha: float = 1e-3
    weights_: Array | None = None
    bias_: Array | None = None

    def fit(self, features: Array, intent: Array) -> RidgeDecoder:
        """Fit decoder weights."""

        features = np.asarray(features, dtype=np.float64)
        intent = np.asarray(intent, dtype=np.float64)
        if features.ndim != 2 or intent.ndim != 2:
            raise ValueError("features and intent must be matrices")
        if features.shape[0] != intent.shape[0]:
            raise ValueError("features and intent must have the same row count")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        x_mean = features.mean(axis=0)
        y_mean = intent.mean(axis=0)
        x = features - x_mean
        y = intent - y_mean
        gram = x.T @ x + self.alpha * np.eye(features.shape[1])
        self.weights_ = np.linalg.solve(gram, x.T @ y)
        self.bias_ = y_mean - x_mean @ self.weights_
        return self

    def predict(self, features: Array) -> Array:
        """Predict intent from features."""

        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("decoder must be fit before predict")
        features = np.asarray(features, dtype=np.float64)
        return features @ self.weights_ + self.bias_

    def score_mse(self, features: Array, intent: Array) -> float:
        """Return mean squared error against intent labels."""
