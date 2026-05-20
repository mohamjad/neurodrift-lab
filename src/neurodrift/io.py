"""Local file IO for reproducible NeuroDrift experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurodrift.session import SessionBatch, SessionPair


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_session_pair_npz(pair: SessionPair, path: Path) -> None:
    """Save a session pair to a compact NPZ artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_neural=pair.source.neural,
        source_intent=pair.source.intent,
        target_neural=pair.target.neural,
        target_intent=pair.target.intent,
        source_session_id=pair.source.session_id,
        target_session_id=pair.target.session_id,
        source_sample_rate_hz=pair.source.sample_rate_hz,
        target_sample_rate_hz=pair.target.sample_rate_hz,
    )


def load_session_pair_npz(path: Path) -> SessionPair:
    """Load a session pair saved by ``save_session_pair_npz``."""

    with np.load(path, allow_pickle=False) as data:
        return SessionPair(
            source=SessionBatch(
                neural=data["source_neural"],
                intent=data["source_intent"],
                session_id=str(data["source_session_id"]),
                sample_rate_hz=float(data["source_sample_rate_hz"]),
            ),
            target=SessionBatch(
                neural=data["target_neural"],
