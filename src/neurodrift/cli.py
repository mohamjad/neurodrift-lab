"""Command line entry point for NeuroDrift experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neurodrift.benchmarks import ALIGNER_REGISTRY, run_alignment_benchmark
from neurodrift.datasets.milimbeeg import fetch_milimbeeg_sample, load_milimbeeg_pair
from neurodrift.datasets.nlb import load_nlb_h5_pair
from neurodrift.envs.intent_drift import IntentDriftEnv
from neurodrift.io import load_session_pair_npz, save_json, save_session_pair_npz
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def _load_config(path: Path | None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return SimulationConfig(**data)


def _load_pair(args: argparse.Namespace):
    if args.input is not None:
        return load_session_pair_npz(args.input)
    return simulate_session_pair(_load_config(args.config))


