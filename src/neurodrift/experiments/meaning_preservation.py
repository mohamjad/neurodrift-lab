"""Experiments where alignment quality is judged by meaning preservation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neurodrift.benchmarks import ALIGNER_REGISTRY
from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.intent import (
    intent_distribution_distance,
    intent_distribution_from_vectors,
    mean_distribution_entropy,
    trial_intent_probabilities,
)
from neurodrift.models.decoders import RidgeDecoder
from neurodrift.plasticity import build_plasticity_signal
from neurodrift.session import SessionPair


@dataclass(frozen=True)
class AmbiguityAudit:
    """Information lost when weak intent labels are collapsed to hard labels."""

    mean_entropy: float
    max_entropy: float
    uncertain_trial_rate: float
    hard_label_loss: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MeaningPreservationRow:
    """Alignment result scored against target-session weak intent semantics."""

    name: str
    target_mse_raw: float
    target_mse_aligned: float
    alignment_gain: float
    raw_meaning_distance: float
    aligned_meaning_distance: float
    meaning_gain: float
    meaning_gap: float
    stability_score: float

    @property
    def preserves_meaning(self) -> bool:
        return self.meaning_gain >= 0.0 and self.aligned_meaning_distance <= 0.25

    @property
    def overaligns(self) -> bool:
        return self.alignment_gain > 0.0 and self.meaning_gain < 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preserves_meaning"] = self.preserves_meaning
        payload["overaligns"] = self.overaligns
        return payload


@dataclass(frozen=True)
class MeaningPreservationReport:
    """Research-facing report for the central NeuroDrift thesis."""

    rows: tuple[MeaningPreservationRow, ...]
    source_target_intent_shift: float
    ambiguity: AmbiguityAudit

    @property
    def best_by_mse(self) -> MeaningPreservationRow:
        return min(self.rows, key=lambda row: row.target_mse_aligned)

    @property
    def best_by_meaning(self) -> MeaningPreservationRow:
        return min(self.rows, key=lambda row: row.aligned_meaning_distance)

    @property
    def has_alignment_meaning_split(self) -> bool:
        return self.best_by_mse.name != self.best_by_meaning.name or any(
            row.overaligns for row in self.rows
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": (
                "alignment should be scored by target-session meaning preservation, "
                "not only decoder error"
            ),
            "source_target_intent_shift": self.source_target_intent_shift,
            "ambiguity": self.ambiguity.to_dict(),
            "best_by_mse": self.best_by_mse.name,
            "best_by_meaning": self.best_by_meaning.name,
            "has_alignment_meaning_split": self.has_alignment_meaning_split,
            "rows": [row.to_dict() for row in self.rows],
        }


def audit_weak_label_ambiguity(pair: SessionPair, temperature: float = 1.0) -> AmbiguityAudit:
    """Measure how much trial-level uncertainty hard labels would erase."""

    probabilities = trial_intent_probabilities(pair.target.intent, temperature=temperature)
    entropy = -(np.maximum(probabilities, 1e-12) * np.log2(np.maximum(probabilities, 1e-12))).sum(
        axis=1
    )
    max_entropy = float(np.log2(probabilities.shape[1]))
    normalized = entropy / max(max_entropy, 1e-8)
    return AmbiguityAudit(
        mean_entropy=float(entropy.mean()),
        max_entropy=max_entropy,
        uncertain_trial_rate=float(np.mean(normalized >= 0.6)),
        hard_label_loss=mean_distribution_entropy(probabilities) / max(max_entropy, 1e-8),
    )


def run_meaning_preservation_experiment(
    pair: SessionPair,
    aligner_names: tuple[str, ...] | None = None,
    temperature: float = 0.75,
) -> MeaningPreservationReport:
    """Run the thesis experiment on a paired BCI session."""

    names = aligner_names or tuple(ALIGNER_REGISTRY)
    unknown = sorted(set(names) - set(ALIGNER_REGISTRY))
    if unknown:
        raise ValueError(f"unknown aligners: {', '.join(unknown)}")

    source_intent = intent_distribution_from_vectors(pair.source.intent, temperature=temperature)
    target_intent = intent_distribution_from_vectors(pair.target.intent, temperature=temperature)
    source_target_shift = intent_distribution_distance(source_intent, target_intent)

    source_x = pair.source.trial_features
    target_x = pair.target.trial_features
    decoder = RidgeDecoder(alpha=1e-2).fit(source_x, pair.source.intent)
    raw_pred = decoder.predict(target_x)
    raw_dist = intent_distribution_distance(
        target_intent,
        intent_distribution_from_vectors(raw_pred, temperature=temperature),
    )
    target_mse_raw = float(np.mean((raw_pred - pair.target.intent) ** 2))

    rows = []
    for name in names:
        aligner = ALIGNER_REGISTRY[name]().fit(source_x, target_x)
        aligned_x = aligner.transform(target_x)
        aligned_pred = decoder.predict(aligned_x)
        aligned_dist = intent_distribution_distance(
            target_intent,
            intent_distribution_from_vectors(aligned_pred, temperature=temperature),
        )
        target_mse_aligned = float(np.mean((aligned_pred - pair.target.intent) ** 2))
        alignment_gain = (target_mse_raw - target_mse_aligned) / max(target_mse_raw, 1e-8)
        result = IntentDriftEnv(pair).evaluate(ALIGNER_REGISTRY[name]())
        signal = build_plasticity_signal(result.report, source_intent, target_intent)
        rows.append(
            MeaningPreservationRow(
                name=name,
                target_mse_raw=target_mse_raw,
                target_mse_aligned=target_mse_aligned,
                alignment_gain=float(alignment_gain),
                raw_meaning_distance=raw_dist,
                aligned_meaning_distance=aligned_dist,
                meaning_gain=raw_dist - aligned_dist,
                meaning_gap=signal.meaning_gap,
                stability_score=result.report.stability_score,
            )
        )

    return MeaningPreservationReport(
        rows=tuple(rows),
        source_target_intent_shift=source_target_shift,
        ambiguity=audit_weak_label_ambiguity(pair, temperature=temperature),
    )
