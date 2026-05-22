"""Small deterministic statistics used by evidence reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ConfidenceInterval:
    """Percentile bootstrap confidence interval."""

    low: float
    high: float
    level: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ScalarSummary:
    """Compact scalar summary for repeated experiments."""

    n: int
    mean: float
    std: float
    ci: ConfidenceInterval

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci": self.ci.to_dict(),
        }


def bootstrap_interval(
    values: Array,
    statistic: Callable[[Array], float] | None = None,
    *,
    samples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Return a deterministic bootstrap interval for one scalar statistic."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be a vector")
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    statistic = statistic or _mean
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for idx in range(samples):
        sample = values[rng.integers(0, values.size, size=values.size)]
        draws[idx] = statistic(sample)
    alpha = 1.0 - level
    return ConfidenceInterval(
        low=float(np.quantile(draws, alpha / 2.0)),
        high=float(np.quantile(draws, 1.0 - alpha / 2.0)),
        level=level,
    )


def summarize(values: Array, *, seed: int = 0) -> ScalarSummary:
    """Mean, sample standard deviation, and bootstrap CI."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be a vector")
    if values.size == 0:
        raise ValueError("values must be non-empty")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return ScalarSummary(
        n=int(values.size),
        mean=float(values.mean()),
        std=std,
        ci=bootstrap_interval(values, seed=seed),
    )


def paired_difference(left: Array, right: Array, *, seed: int = 0) -> ScalarSummary:
    """Summarize paired ``left - right`` values."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired arrays must have the same shape")
    return summarize(left - right, seed=seed)


def cohens_d(values: Array) -> float:
    """One-sample Cohen's d against zero."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be a vector")
    if values.size < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(values.mean() / std)


def _mean(values: Array) -> float:
    return float(values.mean())
