from __future__ import annotations

import numpy as np

from neurodrift.datasets.nlb import load_nlb_h5_pair


def test_nlb_h5_loader_builds_even_odd_pair(tmp_path) -> None:
    import h5py

