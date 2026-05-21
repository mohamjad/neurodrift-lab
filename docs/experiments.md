# Experiments

## Single alignment run

```powershell
neurodrift simulate --config configs\simulated_medium_drift.json --aligner procrustes
```

Fits source. Aligns target. Emits JSON.

## Alignment benchmark

```powershell
neurodrift benchmark --config configs\simulated_high_drift.json
```

Runs all registered aligners on the same pair.

## Reading the results

- `target_mse_aligned`: lower is better.
- `alignment_gain`: positive means adaptation improved target decoding.
- `stability_score`: compact score combining geometry and decoder drift.
- `affine_invariant`: SPD covariance movement between sessions.
- `procrustes_error`: residual after best paired orthogonal alignment.

Simulator presets are for machinery checks. Real sessions should come through
loaders.

## Real-data smoke test

Included EEG fixture:

```powershell
neurodrift benchmark --input data\fixtures\milimbeeg_s1.npz
```

Refresh:

```powershell
neurodrift fetch-milimbeeg --dest data\external\milimbeeg --subject S1 --output data\fixtures\milimbeeg_s1.npz
```

Imagery is source. Motor execution is target. Not invasive, but useful for real
multi-session smoke tests.

## NLB neural-population conversion

If you have the NLB HDF5 evaluation file:

```powershell
neurodrift convert-nlb --input path\to\eval_data_test.h5 --dataset mc_maze_small_20 --output data\fixtures\nlb_mc_maze_small_20.npz
neurodrift benchmark --input data\fixtures\nlb_mc_maze_small_20.npz
```

The adapter splits held-out trials into even/odd pseudo-sessions. Real
population data, smoke-test framing.
