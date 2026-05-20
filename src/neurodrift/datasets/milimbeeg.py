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
    name = f"{subject}R{run}{mode}{task}_{repeat}.csv"
    return f"{MILIMBEEG_RAW_BASE_URL}/{subject}/{name}"


def fetch_milimbeeg_sample(
    dest: Path,
    *,
    subject: str = "S1",
    tasks: tuple[int, ...] = DEFAULT_TASKS,
    repeats: tuple[int, ...] = DEFAULT_REPEATS,
    modes: tuple[str, ...] = ("I", "M"),
    timeout_s: float = 30.0,
) -> list[Path]:
    """Download a small public MILimbEEG subset for real-data smoke tests."""

    dest = Path(dest)
    subject_dir = dest / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for mode in modes:
        for task in tasks:
            for repeat in repeats:
                url = _trial_url(subject, mode, task, repeat)
                path = subject_dir / url.rsplit("/", 1)[-1]
                if not path.exists():
                    try:
                        with urllib.request.urlopen(url, timeout=timeout_s) as response:
                            path.write_bytes(response.read())
                    except (HTTPError, URLError, TimeoutError) as exc:
                        raise RuntimeError(f"failed to fetch {url}") from exc
                downloaded.append(path)
    return downloaded


def _read_trial_csv(path: Path) -> Array:
