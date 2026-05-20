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
neurodrift benchmark --config configs\simulated_high_drift.json
```

This runs every registered alignment strategy against the same session pair and
returns a ranked machine-readable report.

## Reading the results

- `target_mse_aligned`: lower is better.
- `alignment_gain`: positive means adaptation improved target decoding.
- `stability_score`: compact score combining geometry and decoder drift.
- `affine_invariant`: SPD covariance movement between sessions.
- `procrustes_error`: residual after best paired orthogonal alignment.

Use the simulator presets to test the machinery. Replace the simulator with
real session loaders when invasive BCI data is available.

## Real-data smoke test

The repo includes a small converted MILimbEEG/OpenBCI subject fixture:

```powershell
neurodrift benchmark --input data\fixtures\milimbeeg_s1.npz
```

To refresh it from the public CSV source:

```powershell
neurodrift fetch-milimbeeg --dest data\external\milimbeeg --subject S1 --output data\fixtures\milimbeeg_s1.npz
```

The loader treats imagery trials as the source session and motor execution
trials as the target session. This is not invasive BCI, but it is a useful real
multi-session EEG smoke test for the same drift pipeline.

