"""Optional pyRiemann bridge.

Core NeuroDrift metrics are implemented directly to keep the package usable
without optional dependencies. When pyRiemann is installed, this module gives a
clear interop point for downstream experiments.
"""

from __future__ import annotations

import numpy as np
