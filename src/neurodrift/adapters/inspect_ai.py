"""Thin adapter shape for future Inspect AI tasks.

The repo does not require Inspect AI as a dependency. This module keeps the
boundary explicit so an evaluation task can import NeuroDrift without dragging
frontier-eval tooling into core numerical tests.
