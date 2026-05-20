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
