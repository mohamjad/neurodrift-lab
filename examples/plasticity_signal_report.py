"""Show the meaning-gap layer on top of neural drift."""

from __future__ import annotations

import json

from neurodrift.intent import IntentDistribution
from neurodrift.metrics.composite import build_drift_report
from neurodrift.plasticity import build_plasticity_signal
from neurodrift.simulation import SimulationConfig, simulate_session_pair

import numpy as np


def main() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=19, drift_strength=0.32))
    report = build_drift_report(pair)
    source_intent = IntentDistribution(("reach", "rest"), np.array([0.88, 0.12]))
    target_intent = IntentDistribution(("reach", "rest"), np.array([0.58, 0.42]))
    signal = build_plasticity_signal(report, source_intent, target_intent)
    print(
        json.dumps(
            {
                "neural_geometry_shift": signal.neural_geometry_shift,
                "latent_shift": signal.latent_shift,
                "intent_shift": signal.intent_shift,
                "ambiguity": signal.ambiguity,
                "meaning_gap": signal.meaning_gap,
                "needs_relabeling": signal.needs_relabeling,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
