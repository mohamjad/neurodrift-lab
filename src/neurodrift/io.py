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
