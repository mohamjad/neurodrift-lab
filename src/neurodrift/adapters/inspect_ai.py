"""Thin adapter shape for future Inspect AI tasks.

The repo does not require Inspect AI as a dependency. This module keeps the
boundary explicit so an evaluation task can import NeuroDrift without dragging
frontier-eval tooling into core numerical tests.
"""

from __future__ import annotations

from typing import Any

from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.models.alignment import Aligner, WhiteningColoringAligner
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def build_inspect_ready_sample(config: SimulationConfig | None = None) -> dict[str, Any]:
    """Return a serializable task sample for an Inspect-style solver/scorer."""

    pair = simulate_session_pair(config or SimulationConfig())
    env = IntentDriftEnv(pair)
    baseline = env.evaluate()
    return {
        "input": "Choose and justify an alignment strategy for the target BCI session.",
        "metadata": {
            "source_session": pair.source.session_id,
            "target_session": pair.target.session_id,
            "baseline": baseline.to_dict(),
            "recommended_action": "whitening_coloring_alignment",
        },
    }


def score_alignment_action(
    aligner: Aligner | None = None,
    config: SimulationConfig | None = None,
) -> float:
    """Return an alignment-gain score for an Inspect-style scorer."""

    pair = simulate_session_pair(config or SimulationConfig())
