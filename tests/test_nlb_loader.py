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

    pair = load_nlb_h5_pair(path, max_trials=10)

    assert pair.source.neural.shape == (5, 5, 4)
    assert pair.target.neural.shape == (5, 5, 4)
    assert pair.source.intent.shape == (5, 2)
    assert pair.source.session_id == "mc_maze_small_20-even"
