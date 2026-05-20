# Architecture

NeuroDrift Lab is organized around one invariant: longitudinal BCI work should
make drift measurable before adaptation is judged useful.

## Data model

`SessionBatch` stores neural activity as `(trials, time, channels)` and intent
labels as `(trials, intent_dims)`. `SessionPair` couples source and target
sessions that share channel and intent dimensions.

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

`IntentDriftEnv` fits a source decoder, applies an alignment strategy to target
features, and reports whether target intent preservation improved. It is small
on purpose: larger eval runners should wrap this API rather than control core
math directly.

## Simulation

The simulator creates paired source and target sessions from shared latent
intent dynamics. Target sessions receive controlled projection rotation,
feature-gain drift, noise, and small intent perturbations. This makes tests and
example reports reproducible while preserving the distinction between toy data
