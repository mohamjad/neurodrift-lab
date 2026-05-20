"""Run a synthetic longitudinal BCI drift report."""

from __future__ import annotations

import json

from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.models.alignment import ProcrustesAligner, WhiteningColoringAligner
from neurodrift.simulation import SimulationConfig, simulate_session_pair

