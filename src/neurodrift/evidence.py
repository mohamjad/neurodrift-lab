"""Evidence-suite orchestration for the NeuroDrift thesis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurodrift.experiments.meaning_preservation import (
    MeaningPreservationReport,
    run_meaning_preservation_experiment,
)
from neurodrift.io import load_session_pair_npz
from neurodrift.session import SessionPair
from neurodrift.simulation import SimulationConfig, simulate_session_pair
from neurodrift.stats import cohens_d, summarize

DEFAULT_FIXTURE_PATHS = (
    Path("data/fixtures/nlb_mc_maze_small_20.npz"),
    Path("data/fixtures/milimbeeg_s1.npz"),
)


@dataclass(frozen=True)
class EvidenceCase:
    """One session-pair source in an evidence suite."""

    name: str
    kind: str
    source: str
    pair: SessionPair


@dataclass(frozen=True)
class EvidenceRun:
    """One completed meaning-preservation run."""

    name: str
    kind: str
    source: str
    best_by_mse: str
    best_by_meaning: str
    has_alignment_meaning_split: bool
    overaligns: bool
    max_overalignment_penalty: float
    ambiguity: float
    uncertain_trial_rate: float
    source_target_intent_shift: float
    best_mse_alignment_gain: float
    best_meaning_distance: float
    report: MeaningPreservationReport

    def to_dict(self, *, include_report: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("report")
        if include_report:
            payload["report"] = self.report.to_dict()
        return payload


@dataclass(frozen=True)
class EvidenceSummary:
    """Aggregate statistics over an evidence suite."""

    suite: str
    runs: tuple[EvidenceRun, ...]
    statistics: dict[str, Any]

    def to_dict(self, *, include_reports: bool = True) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "claim": (
                "decoder performance and target-session intent preservation can diverge "
                "under cross-session neural drift"
            ),
            "n_runs": len(self.runs),
            "statistics": self.statistics,
            "runs": [run.to_dict(include_report=include_reports) for run in self.runs],
        }


def synthetic_evidence_cases(
    *,
    seed_count: int = 24,
    drift_strengths: tuple[float, ...] = (0.12, 0.2, 0.35, 0.5),
) -> tuple[EvidenceCase, ...]:
    """Build controlled cases spanning mild to large session drift."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    cases = []
    for seed in range(1, seed_count + 1):
        for drift_strength in drift_strengths:
            config = SimulationConfig(
                seed=seed,
                drift_strength=drift_strength,
                noise_scale=0.05 if drift_strength <= 0.2 else 0.06,
            )
            name = f"synthetic_seed_{seed:03d}_drift_{drift_strength:.2f}"
            cases.append(
                EvidenceCase(
                    name=name,
                    kind="synthetic",
                    source=f"SimulationConfig(seed={seed}, drift_strength={drift_strength:.2f})",
                    pair=simulate_session_pair(config),
                )
            )
    return tuple(cases)


def fixture_evidence_cases(
    fixture_paths: Iterable[Path] = DEFAULT_FIXTURE_PATHS,
) -> tuple[EvidenceCase, ...]:
    """Load local real-data smoke-test fixtures when present."""

    cases = []
    for path in fixture_paths:
        if not path.exists():
            continue
        cases.append(
            EvidenceCase(
                name=path.stem,
                kind="fixture",
                source=str(path),
                pair=load_session_pair_npz(path),
            )
        )
    return tuple(cases)


def run_evidence_suite(
    *,
    suite: str = "synthetic",
    seed_count: int = 24,
    include_fixtures: bool = True,
    temperature: float = 0.75,
) -> EvidenceSummary:
    """Run a repeatable suite of meaning-preservation evidence cases."""

    cases: list[EvidenceCase] = []
    if suite in {"synthetic", "all"}:
        cases.extend(synthetic_evidence_cases(seed_count=seed_count))
    if suite in {"fixtures", "all"} and include_fixtures:
        cases.extend(fixture_evidence_cases())
    if not cases:
        raise ValueError(f"no evidence cases available for suite: {suite}")
    runs = tuple(_run_case(case, temperature=temperature) for case in cases)
    return EvidenceSummary(suite=suite, runs=runs, statistics=_summarize_runs(runs))


def _run_case(case: EvidenceCase, *, temperature: float) -> EvidenceRun:
    report = run_meaning_preservation_experiment(case.pair, temperature=temperature)
    best_mse = report.best_by_mse
    best_meaning = report.best_by_meaning
    overaligning_rows = [row for row in report.rows if row.overaligns]
    max_penalty = max((-row.meaning_gain for row in overaligning_rows), default=0.0)
    return EvidenceRun(
        name=case.name,
        kind=case.kind,
        source=case.source,
        best_by_mse=best_mse.name,
        best_by_meaning=best_meaning.name,
        has_alignment_meaning_split=report.has_alignment_meaning_split,
        overaligns=bool(overaligning_rows),
        max_overalignment_penalty=float(max_penalty),
        ambiguity=report.ambiguity.hard_label_loss,
        uncertain_trial_rate=report.ambiguity.uncertain_trial_rate,
        source_target_intent_shift=report.source_target_intent_shift,
        best_mse_alignment_gain=best_mse.alignment_gain,
        best_meaning_distance=best_meaning.aligned_meaning_distance,
        report=report,
    )


def _summarize_runs(runs: tuple[EvidenceRun, ...]) -> dict[str, Any]:
    split = np.asarray([run.has_alignment_meaning_split for run in runs], dtype=np.float64)
    overalign = np.asarray([run.overaligns for run in runs], dtype=np.float64)
    penalties = np.asarray([run.max_overalignment_penalty for run in runs], dtype=np.float64)
    ambiguity = np.asarray([run.ambiguity for run in runs], dtype=np.float64)
    shifts = np.asarray([run.source_target_intent_shift for run in runs], dtype=np.float64)
    gains = np.asarray([run.best_mse_alignment_gain for run in runs], dtype=np.float64)
    meaning = np.asarray([run.best_meaning_distance for run in runs], dtype=np.float64)
    return {
        "split_rate": summarize(split, seed=1).to_dict(),
        "overalignment_rate": summarize(overalign, seed=2).to_dict(),
        "overalignment_penalty": summarize(penalties, seed=3).to_dict(),
        "hard_label_loss": summarize(ambiguity, seed=4).to_dict(),
        "source_target_intent_shift": summarize(shifts, seed=5).to_dict(),
        "best_mse_alignment_gain": summarize(gains, seed=6).to_dict(),
        "best_meaning_distance": summarize(meaning, seed=7).to_dict(),
        "overalignment_penalty_effect_size": cohens_d(penalties),
        "cases_by_kind": _count_by_kind(runs),
    }


def _count_by_kind(runs: tuple[EvidenceRun, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.kind] = counts.get(run.kind, 0) + 1
    return counts
