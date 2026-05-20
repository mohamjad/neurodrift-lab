# Roadmap

## Phase 1: Working core

- Riemannian covariance drift metrics.
- Latent trajectory and subspace drift metrics.
- Simulated BCI sessions with controllable drift.
- Baseline decoders and alignment methods.
- Evaluation environment with machine-readable scores.

## Phase 2: Upstream integrations

- Optional pyRiemann-backed implementations.
- Optional geomstats manifolds.
- Inspect AI task wrapper.
- METR task-standard packaging.
- BRAND stream adapter.

## Phase 3: Research workflows

- Public dataset loaders where licensing allows.
- Drift dashboard.
- Reproducible paper-style experiment configs.
- Benchmark cards for adaptive decoders.

## Future repo splits

Only split once APIs stabilize:

- `neural-drift-metrics`: standalone metrics library.
- `bci-intent-drift-gym`: benchmark environments.
- `intent-drift-experiments`: paper artifacts.
- `bci-drift-dashboard`: visualization app.
