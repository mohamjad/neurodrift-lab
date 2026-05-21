# Architecture

The repo is built around one rule: measure drift before calling adaptation
useful.

## Data model

`SessionBatch` is `(trials, time, channels)` neural data plus `(trials,
intent_dims)` labels.

`SessionPair` is the source-target unit every metric consumes.

## Metric stack

- `metrics.riemannian`: covariance geometry on SPD matrices.
- `metrics.subspace`: principal angles and Procrustes errors.
- `metrics.trajectory`: velocity, path length, and curvature movement.
- `metrics.decoder`: output-level intent preservation.
- `metrics.composite`: machine-readable summary for eval runners.

## Model stack

- `RidgeDecoder`: deterministic baseline decoder.
- `IdentityAligner`: no-op baseline.
- `CenteringTransform`: corrects mean shifts.
- `ProcrustesAligner`: paired orthogonal alignment.
- `WhiteningColoringAligner`: covariance-level session alignment.

## Environment stack

`IntentDriftEnv` fits on source, aligns target, scores target intent
preservation, and emits a report. External eval systems can wrap that.

## Simulation

The simulator creates paired sessions from shared latent intent dynamics, then
adds rotation, gain shift, noise, and small intent perturbations.

It is a test harness. Real data comes through dataset loaders.
