from __future__ import annotations

from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.models.alignment import ProcrustesAligner, WhiteningColoringAligner
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_simulated_pair_has_expected_shapes() -> None:
    config = SimulationConfig(trials=32, time_steps=12, channels=7, intent_dims=2, latent_dims=4)
    pair = simulate_session_pair(config)

    assert pair.source.neural.shape == (32, 12, 7)
    assert pair.target.neural.shape == (32, 12, 7)
    assert pair.source.intent.shape == (32, 2)
    assert pair.target.intent.shape == (32, 2)


def test_env_returns_machine_readable_scores() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=8, trials=48, time_steps=18, channels=9))
    result = IntentDriftEnv(pair).evaluate(WhiteningColoringAligner())
    payload = result.to_dict()

    assert 0.0 <= result.report.stability_score <= 1.0
    assert "alignment_gain" in payload
    assert "affine_invariant" in payload["report"]["covariance"]
