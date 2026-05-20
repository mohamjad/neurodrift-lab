# Evaluation Environments

`IntentDriftEnv` is the repo's benchmark nucleus. It keeps the core operation
small enough to audit:

1. Fit a decoder on source-session neural features.
2. Apply an alignment strategy to target-session features.
3. Score target intent preservation.
4. Emit drift metrics that an eval runner can consume.

