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
        self.decoder = decoder or RidgeDecoder(alpha=1e-2)

    def evaluate(self, aligner: Aligner | None = None) -> DriftEvaluationResult:
        """Fit a source decoder and score target-session preservation."""

        aligner = aligner or IdentityAligner()
        source_x = self.pair.source.trial_features
        target_x = self.pair.target.trial_features
        source_y = self.pair.source.intent
        target_y = self.pair.target.intent

        self.decoder.fit(source_x, source_y)
        source_pred = self.decoder.predict(source_x)
        raw_target_pred = self.decoder.predict(target_x)

        aligner.fit(source_x, target_x)
        aligned_target_x = aligner.transform(target_x)
        aligned_target_pred = self.decoder.predict(aligned_target_x)

        source_mse = self.decoder.score_mse(source_x, source_y)
        target_mse_raw = self.decoder.score_mse(target_x, target_y)
        target_mse_aligned = float(((aligned_target_pred - target_y) ** 2).mean())
        alignment_gain = (target_mse_raw - target_mse_aligned) / max(target_mse_raw, 1e-8)

        report = build_drift_report(
            self.pair,
            source_decoder_outputs=source_pred,
            target_decoder_outputs=aligned_target_pred,
        )
        _ = decoder_drift(raw_target_pred, aligned_target_pred)
        return DriftEvaluationResult(
            source_mse=source_mse,
            target_mse_raw=target_mse_raw,
            target_mse_aligned=target_mse_aligned,
            alignment_gain=float(alignment_gain),
