"""Paper-facing experiments for NeuroDrift Lab."""

from neurodrift.experiments.meaning_preservation import (
    AmbiguityAudit,
    MeaningPreservationReport,
    MeaningPreservationRow,
    run_meaning_preservation_experiment,
)

__all__ = [
    "AmbiguityAudit",
    "MeaningPreservationReport",
    "MeaningPreservationRow",
    "run_meaning_preservation_experiment",
]
