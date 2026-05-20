"""Environment-style benchmark for session-to-session intent drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from neurodrift.metrics.composite import DriftReport, build_drift_report
from neurodrift.metrics.decoder import decoder_drift
from neurodrift.models.alignment import Aligner, IdentityAligner
from neurodrift.models.decoders import RidgeDecoder
from neurodrift.session import SessionPair


@dataclass(frozen=True)
class DriftEvaluationResult:
    """Score payload returned by an intent-drift evaluation run."""

    source_mse: float
    target_mse_raw: float
    target_mse_aligned: float
    alignment_gain: float
    report: DriftReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["report"] = self.report.to_dict()
        return payload


class IntentDriftEnv:
    """Small deterministic BCI evaluation environment.

    The environment is intentionally not tied to one RL framework. It exposes a
    stable ``evaluate`` method that can be wrapped by Inspect AI, METR tasks, or
    Gym-style interfaces later.
    """

    def __init__(self, pair: SessionPair, decoder: RidgeDecoder | None = None) -> None:
        self.pair = pair
