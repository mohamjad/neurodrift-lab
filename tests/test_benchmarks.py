from __future__ import annotations

from neurodrift.benchmarks import run_alignment_benchmark
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_alignment_benchmark_ranks_registered_strategies() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=12, trials=64, channels=10))
    benchmark = run_alignment_benchmark(pair)

    assert len(benchmark.rows) == 4
    assert benchmark.best_by_target_mse.name in {row.name for row in benchmark.rows}
    assert benchmark.best_by_gain.name in {row.name for row in benchmark.rows}
    assert "rows" in benchmark.to_dict()


def test_alignment_benchmark_rejects_unknown_strategy() -> None:
    pair = simulate_session_pair(SimulationConfig(seed=13, trials=32, channels=8))

    try:
        run_alignment_benchmark(pair, ("identity", "missing"))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown aligner should raise ValueError")
