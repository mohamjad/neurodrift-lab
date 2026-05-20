# Evaluation Environments

`IntentDriftEnv` is the repo's benchmark nucleus. It keeps the core operation
small enough to audit:

1. Fit a decoder on source-session neural features.
2. Apply an alignment strategy to target-session features.
3. Score target intent preservation.
4. Emit drift metrics that an eval runner can consume.

## Frontier-lab wrapping

The environment can be wrapped by external eval systems without changing core
math:

- Inspect AI: use the adapter boundary in `neurodrift.adapters.inspect_ai`.
- METR task-standard: package a config, fixed seed, command, and scorer around
  `neurodrift benchmark`.
- NeMo Gym or Gymnasium: expose alignment choice as the action and report
  alignment gain or stability score as the reward.

## BCI wrapping

For online BCI systems, replace `simulate_session_pair` with a loader or stream
adapter that returns `SessionPair`. The rest of the evaluation path is unchanged.
