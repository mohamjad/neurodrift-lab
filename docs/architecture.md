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
