from __future__ import annotations

from neurodrift.cli import main


def test_cli_simulate_runs(capsys) -> None:
    exit_code = main(["simulate", "--aligner", "center"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "alignment_gain" in captured.out


def test_cli_benchmark_runs(capsys) -> None:
    exit_code = main(["benchmark"])
