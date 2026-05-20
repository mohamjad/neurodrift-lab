# Experiments

## Single alignment run

```powershell
neurodrift simulate --config configs\simulated_medium_drift.json --aligner procrustes
```

This trains a source-session decoder, applies one target-session alignment
strategy, and emits JSON with target MSE, alignment gain, covariance drift,
trajectory drift, decoder drift, and stability score.

## Alignment benchmark

```powershell
