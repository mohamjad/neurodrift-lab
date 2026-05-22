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

## Thesis experiment

```powershell
neurodrift thesis --config configs\simulated_alignment_meaning_split.json
```

This is the research-facing experiment. It asks whether each alignment method
preserves target-session meaning, not only whether it lowers decoder MSE.

Read:

- `best_by_mse`: method with the lowest aligned target error.
- `best_by_meaning`: method with the lowest weak-intent semantic distance.
- `has_alignment_meaning_split`: true when those two objectives diverge.
- `hard_label_loss`: how much ambiguity disappears under hard labels.

If `alignment_gain` is positive and `meaning_gain` is negative, the method
over-aligned: it made the source decoder look better while moving away from the
target session's weak intent.

## Evidence suite

```powershell
neurodrift evidence --suite synthetic --compact --figure-dir artifacts\evidence
```

This runs repeated thesis experiments across seeds and drift levels. The output
adds bootstrap intervals for split rate, over-alignment rate, ambiguity, and
meaning loss.

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
