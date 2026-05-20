"""Compare alignment strategies on the same synthetic session pair."""

from __future__ import annotations

import json

from neurodrift.benchmarks import run_alignment_benchmark
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def main() -> None:
    config = SimulationConfig(seed=21, drift_strength=0.28, trials=120, channels=20)
    benchmark = run_alignment_benchmark(simulate_session_pair(config))
    print(json.dumps(benchmark.to_dict(), indent=2))


if __name__ == "__main__":
    main()
