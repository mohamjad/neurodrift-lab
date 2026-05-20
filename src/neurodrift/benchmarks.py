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
