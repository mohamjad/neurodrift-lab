"""Run a synthetic longitudinal BCI drift report."""

from __future__ import annotations

import json

from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.models.alignment import ProcrustesAligner, WhiteningColoringAligner
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def main() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=11, drift_strength=0.22))
    env = IntentDriftEnv(pair)
    baseline = env.evaluate(ProcrustesAligner())
    covariance_aligned = env.evaluate(WhiteningColoringAligner())
    print(
        json.dumps(
            {
                "procrustes": baseline.to_dict(),
                "whitening_coloring": covariance_aligned.to_dict(),
            },
            indent=2,
        )
    )
