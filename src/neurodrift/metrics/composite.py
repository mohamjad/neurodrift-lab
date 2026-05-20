"""Composite reporting across geometry, latent movement, and decoder behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from neurodrift.metrics.decoder import DecoderDrift, decoder_drift
from neurodrift.metrics.riemannian import CovarianceDrift, covariance_drift
from neurodrift.metrics.subspace import procrustes_error, subspace_distance
from neurodrift.metrics.trajectory import TrajectoryDrift, trajectory_drift
from neurodrift.session import SessionPair


@dataclass(frozen=True)
class DriftReport:
    """End-to-end drift report for one session pair."""

    source_session: str
    target_session: str
    covariance: CovarianceDrift
    trajectory: TrajectoryDrift
    decoder: DecoderDrift | None
    subspace_distance: float
    procrustes_error: float
    stability_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar report values for logs, JSON, and eval scorers."""

        payload = asdict(self)
        payload["covariance"].pop("source_covariance", None)
        payload["covariance"].pop("target_covariance", None)
        return payload


def build_drift_report(
    pair: SessionPair,
    *,
    source_decoder_outputs=None,
    target_decoder_outputs=None,
) -> DriftReport:
    """Build a scalar drift report from paired source and target sessions."""

    covariance = covariance_drift(pair.source.flattened_neural, pair.target.flattened_neural)
