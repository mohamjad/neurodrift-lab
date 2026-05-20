from __future__ import annotations

import numpy as np

from neurodrift.datasets.nlb import load_nlb_h5_pair


def test_nlb_h5_loader_builds_even_odd_pair(tmp_path) -> None:
    import h5py

    path = tmp_path / "nlb.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("mc_maze_small_20")
        group.create_dataset("eval_spikes_heldout", data=np.ones((12, 5, 4)))
        group.create_dataset("eval_behavior", data=np.ones((12, 5, 2)))
