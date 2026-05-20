"""Alignment transforms for adapting target sessions to a source decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

