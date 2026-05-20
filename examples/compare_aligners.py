"""Compare alignment strategies on the same synthetic session pair."""

from __future__ import annotations

import json

from neurodrift.benchmarks import run_alignment_benchmark
from neurodrift.simulation import SimulationConfig, simulate_session_pair


