"""Run the central NeuroDrift meaning-preservation experiment."""

from __future__ import annotations

import json

from neurodrift.experiments.meaning_preservation import run_meaning_preservation_experiment
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def main() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=17, drift_strength=0.42, noise_scale=0.06))
    report = run_meaning_preservation_experiment(pair)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
