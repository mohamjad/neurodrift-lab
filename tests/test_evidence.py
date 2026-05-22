from __future__ import annotations

from neurodrift.evidence import run_evidence_suite, synthetic_evidence_cases
from neurodrift.figures import render_mse_vs_meaning, render_split_rate


def test_synthetic_evidence_cases_are_named_and_bounded() -> None:
    cases = synthetic_evidence_cases(seed_count=2, drift_strengths=(0.1, 0.2))

    assert len(cases) == 4
    assert cases[0].kind == "synthetic"
    assert "drift" in cases[0].name


def test_evidence_suite_summarizes_repeated_runs() -> None:
    summary = run_evidence_suite(suite="synthetic", seed_count=2)

    assert summary.statistics["split_rate"]["n"] == 8
    assert summary.statistics["cases_by_kind"] == {"synthetic": 8}
    assert "decoder performance" in summary.to_dict(include_reports=False)["claim"]


def test_evidence_figures_render_svg() -> None:
    summary = run_evidence_suite(suite="synthetic", seed_count=1)

    assert render_split_rate(summary).startswith("<svg")
    assert "circle" in render_mse_vs_meaning(summary)
