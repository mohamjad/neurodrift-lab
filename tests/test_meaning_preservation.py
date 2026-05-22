from __future__ import annotations

import numpy as np

from neurodrift.experiments.meaning_preservation import run_meaning_preservation_experiment
from neurodrift.intent import intent_distribution_from_vectors, trial_intent_probabilities
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_intent_distribution_from_vectors_is_stable() -> None:
    vectors = np.array([[2.0, 0.1], [1.8, 0.2], [1.6, 0.3]])
    distribution = intent_distribution_from_vectors(vectors, ("reach", "rest"))
    signed = intent_distribution_from_vectors(vectors)

    assert distribution.labels == ("reach", "rest")
    assert distribution.probabilities[0] > distribution.probabilities[1]
    assert signed.labels == ("intent_0_pos", "intent_0_neg", "intent_1_pos", "intent_1_neg")


def test_trial_intent_probabilities_preserve_uncertainty_shape() -> None:
    vectors = np.array([[1.0, 1.0], [3.0, 0.1]])
    probabilities = trial_intent_probabilities(vectors)

    assert probabilities.shape == (2, 2)
    assert probabilities[0].max() < probabilities[1].max()


def test_meaning_preservation_experiment_scores_alignment_semantics() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=11, drift_strength=0.35))
    report = run_meaning_preservation_experiment(pair)

    assert len(report.rows) == 4
    assert report.best_by_mse.name in {row.name for row in report.rows}
    assert report.best_by_meaning.name in {row.name for row in report.rows}
    assert report.ambiguity.hard_label_loss > 0.0
    assert "rows" in report.to_dict()


def test_meaning_preservation_can_split_from_decoder_mse() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=1, drift_strength=0.2, noise_scale=0.05))
    report = run_meaning_preservation_experiment(pair)

    assert report.has_alignment_meaning_split
    assert report.best_by_mse.name != report.best_by_meaning.name
