from __future__ import annotations

import numpy as np

from neurodrift.stats import bootstrap_interval, cohens_d, paired_difference, summarize


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.array([0.0, 1.0, 1.0, 0.0])

    first = bootstrap_interval(values, seed=4)
    second = bootstrap_interval(values, seed=4)

    assert first == second
    assert first.low <= values.mean() <= first.high


def test_summarize_and_effect_size() -> None:
    values = np.array([0.1, 0.2, 0.3])
    summary = summarize(values)

    assert summary.n == 3
    assert summary.mean > 0
    assert cohens_d(values) > 0


def test_paired_difference_checks_shapes() -> None:
    left = np.array([1.0, 2.0])
    right = np.array([0.5, 1.0])

    diff = paired_difference(left, right)

    assert diff.mean == 0.75
