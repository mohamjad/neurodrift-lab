# NeuroDrift Lab

NeuroDrift Lab is a research-grade monorepo for measuring session-to-session
intent drift, neural manifold movement, and decoder stability in BCI-style
systems.

The project is deliberately built as a composable lab rather than a one-off
notebook. It includes:

- Riemannian metrics for covariance and latent-state drift.
- Session simulation for controlled neuroplasticity and intent-drift studies.
- Decoder and alignment baselines.
- A small evaluation environment API that can later be wrapped by Inspect AI,
  METR task-standard tasks, NeMo Gym, or Gymnasium.
- Tests and examples intended to keep the code readable and auditable.

## Why this exists

Most BCI repos focus on decoding performance within a session. This repo treats
longitudinal drift as the first-class object:

- How far did the neural manifold move?
- Did decoder outputs preserve intent after the move?
- Which alignment strategy best stabilizes the interface?
- Can we score an adaptive decoder as an environment, not just a static model?

## Status

This is an early but working foundation. It does not vendor upstream projects.
Instead, it exposes clean adapters and documents how to integrate established
tools such as pyRiemann, Geomstats, LFADS, NoMAD-style latent alignment,
BRAND, and BCI simulators.

## Quickstart

