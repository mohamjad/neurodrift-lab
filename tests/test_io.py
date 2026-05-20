from __future__ import annotations

import json

import numpy as np

from neurodrift.io import load_session_pair_npz, save_json, save_session_pair_npz
from neurodrift.simulation import SimulationConfig, simulate_session_pair


def test_session_pair_npz_roundtrip(tmp_path) -> None:
    pair = simulate_session_pair(SimulationConfig(seed=14, trials=16, channels=6))
    path = tmp_path / "pair.npz"

    save_session_pair_npz(pair, path)
