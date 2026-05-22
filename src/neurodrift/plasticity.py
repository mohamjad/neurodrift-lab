"""Plasticity-as-signal reporting."""

from __future__ import annotations

from dataclasses import dataclass

from neurodrift.intent import IntentDistribution, intent_distribution_distance
from neurodrift.metrics.composite import DriftReport


@dataclass(frozen=True)
class PlasticitySignal:
    """Joint neural and intent movement summary."""

    neural_geometry_shift: float
    latent_shift: float
    intent_shift: float
    ambiguity: float
    meaning_gap: float

    @property
    def needs_relabeling(self) -> bool:
        return self.meaning_gap > 0.45 or self.ambiguity > 1.0


def build_plasticity_signal(
    report: DriftReport,
    source_intent: IntentDistribution,
    target_intent: IntentDistribution,
) -> PlasticitySignal:
    """Combine neural drift and weak intent drift into one inspectable signal."""

    intent_shift = intent_distribution_distance(source_intent, target_intent)
    ambiguity = max(source_intent.entropy, target_intent.entropy)
    neural_geometry_shift = report.covariance.affine_invariant
    latent_shift = report.trajectory.mean_position_shift
    meaning_gap = (
        0.35 * min(neural_geometry_shift / 10.0, 1.0)
        + 0.25 * min(latent_shift, 1.0)
        + 0.25 * min(intent_shift, 1.0)
        + 0.15 * min(ambiguity / 2.0, 1.0)
    )
    return PlasticitySignal(
        neural_geometry_shift=neural_geometry_shift,
        latent_shift=latent_shift,
        intent_shift=intent_shift,
        ambiguity=ambiguity,
        meaning_gap=meaning_gap,
    )
