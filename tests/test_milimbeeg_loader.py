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
