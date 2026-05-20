"""MILimbEEG/OpenBCI loader for real multi-session EEG smoke tests."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
