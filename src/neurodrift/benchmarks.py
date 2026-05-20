"""Benchmark orchestration for comparing adaptation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neurodrift.envs.intent_drift import DriftEvaluationResult, IntentDriftEnv
from neurodrift.models.alignment import (
    CenteringTransform,
    IdentityAligner,
    ProcrustesAligner,
    WhiteningColoringAligner,
)
from neurodrift.session import SessionPair

ALIGNER_REGISTRY = {
    "identity": IdentityAligner,
    "center": CenteringTransform,
    "procrustes": ProcrustesAligner,
    "whiten-color": WhiteningColoringAligner,
}


@dataclass(frozen=True)
class AlignmentBenchmarkRow:
    """One scored adaptation strategy."""

    name: str
    result: DriftEvaluationResult

    @property
    def target_mse(self) -> float:
        return self.result.target_mse_aligned

    @property
    def alignment_gain(self) -> float:
        return self.result.alignment_gain

    @property
    def stability_score(self) -> float:
        return self.result.report.stability_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_mse": self.target_mse,
            "alignment_gain": self.alignment_gain,
            "stability_score": self.stability_score,
            "result": self.result.to_dict(),
