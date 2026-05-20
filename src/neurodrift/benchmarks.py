"""Benchmark orchestration for comparing adaptation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neurodrift.envs.intent_drift import DriftEvaluationResult, IntentDriftEnv
from neurodrift.models.alignment import (
    CenteringTransform,
