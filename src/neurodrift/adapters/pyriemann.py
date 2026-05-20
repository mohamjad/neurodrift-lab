"""Optional pyRiemann bridge.

Core NeuroDrift metrics are implemented directly to keep the package usable
without optional dependencies. When pyRiemann is installed, this module gives a
clear interop point for downstream experiments.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def pyriemann_distance(source_covariance: Array, target_covariance: Array) -> float:
    """Compute pyRiemann's Riemannian distance if the dependency is installed."""

    try:
        from pyriemann.utils.distance import distance_riemann
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install neurodrift-lab[integrations] to use pyRiemann") from exc
    return float(distance_riemann(source_covariance, target_covariance))
