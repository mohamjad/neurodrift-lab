from __future__ import annotations

import numpy as np

from neurodrift.datasets.milimbeeg import load_milimbeeg_pair


def _write_trial(path, value: float) -> None:
    rows = np.column_stack(
        [
