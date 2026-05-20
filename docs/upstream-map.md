# Upstream Integration Map

NeuroDrift Lab is not a fork of any single project. It is a systems layer that
connects established BCI, geometry, and frontier-eval ideas.

## Geometry and BCI signal processing

- `pyRiemann`: reference implementation for SPD covariance geometry,
  tangent-space features, MDM classifiers, and Procrustes alignment utilities.
- `geomstats`: broader Riemannian geometry toolkit for custom manifolds.

## Neural latent dynamics

- NoMAD: strong reference for stabilizing BCI with latent dynamics alignment.
- LFADS and Neural Data Transformers: upstream-style tools for extracting
  denoised latent dynamics from neural populations.
- MARBLE: geometric deep learning approach for latent vector fields.
- Neural Latents Benchmark: real neural population HDF5 data can be converted
  through `neurodrift convert-nlb` for motor-cortex smoke tests.

## Online BCI systems

- BRAND: real-time asynchronous neural decoding backend.
- BCI simulator repos: useful for closed-loop replay and decoder stress tests.
- MILimbEEG/OpenBCI: public multi-session EEG CSV recordings used here as a
  real-data smoke test for session-pair loading and benchmarking.

## Frontier evaluation environments

- Inspect AI: clean solver/scorer abstraction for model evaluation.
- METR task-standard: containerized agent tasks with machine-verifiable scoring.
- NeMo Gym and SWE/terminal benchmarks: environment-first evaluation patterns.

## Policy

