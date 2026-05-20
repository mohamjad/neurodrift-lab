from __future__ import annotations

from neurodrift.benchmarks import run_alignment_benchmark
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_alignment_benchmark_ranks_registered_strategies() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=12, trials=64, channels=10))
    benchmark = run_alignment_benchmark(pair)

