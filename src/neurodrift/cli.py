"""Command line entry point for NeuroDrift experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neurodrift.benchmarks import ALIGNER_REGISTRY, run_alignment_benchmark
