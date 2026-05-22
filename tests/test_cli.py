from __future__ import annotations

from neurodrift.cli import main


def test_cli_simulate_runs(capsys) -> None:
    exit_code = main(["simulate", "--aligner", "center"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "alignment_gain" in captured.out


def test_cli_benchmark_runs(capsys) -> None:
    exit_code = main(["benchmark"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "best_by_target_mse" in captured.out


def test_cli_thesis_runs(capsys) -> None:
    exit_code = main(["thesis"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "best_by_meaning" in captured.out


def test_cli_evidence_runs(capsys) -> None:
    exit_code = main(["evidence", "--suite", "synthetic", "--seed-count", "1", "--compact"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "split_rate" in captured.out
