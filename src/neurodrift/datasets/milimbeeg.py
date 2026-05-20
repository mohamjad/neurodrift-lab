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
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"expected sampling column plus EEG channels in {path}")
    return data[:, 1:]


def _trial_metadata(path: Path) -> tuple[str, int, str, int]:
    match = _FILE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse MILimbEEG trial filename: {path.name}")
    return (
        match.group("subject"),
        int(match.group("run")),
        match.group("mode"),
        int(match.group("task")),
    )


def _one_hot(labels: list[int]) -> Array:
    unique = sorted(set(labels))
    index = {label: idx for idx, label in enumerate(unique)}
    encoded = np.zeros((len(labels), len(unique)), dtype=np.float64)
    for row, label in enumerate(labels):
        encoded[row, index[label]] = 1.0
    return encoded


def _load_mode_trials(subject_dir: Path, mode: str, tasks: tuple[int, ...]) -> tuple[Array, Array]:
    paths = sorted(subject_dir.glob(f"*{mode}*.csv"))
    trials: list[Array] = []
    labels: list[int] = []
    for path in paths:
        _, _, parsed_mode, task = _trial_metadata(path)
        if parsed_mode == mode and task in tasks:
            trials.append(_read_trial_csv(path))
            labels.append(task)
    if not trials:
        raise ValueError(f"no MILimbEEG {mode} trials found in {subject_dir}")
    min_time = min(trial.shape[0] for trial in trials)
    neural = np.stack([trial[:min_time] for trial in trials]).astype(np.float64)
