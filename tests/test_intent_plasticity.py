from __future__ import annotations

import numpy as np

from neurodrift.intent import (
    IntentDistribution,
    intent_distribution_distance,
    soft_intent_from_observations,
)
from neurodrift.metrics.composite import build_drift_report
from neurodrift.plasticity import build_plasticity_signal
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_soft_intent_distribution_tracks_ambiguity() -> None:
    labels = ("left", "right")
    sharp = soft_intent_from_observations(labels, {"left": 2.0, "right": 0.0})
    ambiguous = soft_intent_from_observations(labels, {"left": 1.0, "right": 1.0})

    assert sharp.confidence > ambiguous.confidence
    assert ambiguous.entropy > sharp.entropy


def test_intent_distribution_distance_detects_meaning_shift() -> None:
    labels = ("speak", "rest")
    source = IntentDistribution(labels, np.array([0.9, 0.1]))
    target = IntentDistribution(labels, np.array([0.2, 0.8]))

    assert intent_distribution_distance(source, target) > 0.25


def test_plasticity_signal_combines_neural_and_intent_shift() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=42, drift_strength=0.3))
    report = build_drift_report(pair)
    source = IntentDistribution(("reach", "rest"), np.array([0.85, 0.15]))
    target = IntentDistribution(("reach", "rest"), np.array([0.55, 0.45]))

    signal = build_plasticity_signal(report, source, target)

    assert signal.meaning_gap > 0.0
    assert signal.neural_geometry_shift == report.covariance.affine_invariant
