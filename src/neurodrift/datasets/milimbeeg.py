"""MILimbEEG/OpenBCI loader for real multi-session EEG smoke tests."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
from numpy.typing import NDArray

from neurodrift.session import SessionBatch, SessionPair

Array = NDArray[np.float64]
MILIMBEEG_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/vasanza/BCI_Motor_Imagery_Task_OpenBCI"
    "/main/MatlabCode/data"
)
DEFAULT_TASKS = (2, 3, 4, 5, 6, 7)
DEFAULT_REPEATS = (1, 2, 3)
_FILE_RE = re.compile(r"(?P<subject>S\d+)R(?P<run>\d+)(?P<mode>[IM])(?P<task>\d+)_")


def _trial_url(subject: str, mode: str, task: int, repeat: int, run: int = 1) -> str:
