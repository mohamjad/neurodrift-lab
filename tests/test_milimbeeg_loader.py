from __future__ import annotations

import numpy as np

from neurodrift.datasets.milimbeeg import load_milimbeeg_pair


def _write_trial(path, value: float) -> None:
    rows = np.column_stack(
        [
            np.arange(8),
            np.full((8, 3), value, dtype=np.float64) + np.arange(8)[:, None] * 0.1,
        ]
    )
    header = ",0,1,2"
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def test_milimbeeg_loader_builds_session_pair(tmp_path) -> None:
    subject_dir = tmp_path / "S1"
    subject_dir.mkdir()
    for mode, offset in (("I", 0.0), ("M", 1.0)):
        for task in (2, 3):
            for repeat in (1, 2):
                _write_trial(subject_dir / f"S1R1{mode}{task}_{repeat}.csv", offset + task)
