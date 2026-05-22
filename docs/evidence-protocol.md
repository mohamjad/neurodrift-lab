# Evidence Protocol

The central claim is narrow:

```text
decoder performance and target-session intent preservation can diverge under cross-session neural drift
```

The repo treats that as an evaluation problem.

## Anchors

This builds on work the field already respects:

- manifold stabilization for long-term BCI control
- adaptive decoding under neural non-stationarity
- covariate-shift adaptation
- uncertainty-aware neural decoding
- neural latent benchmarks and public NWB/DANDI datasets

The distinction is the measured object. NeuroDrift does not ask only whether an
alignment method improves control. It asks whether the target session's weak
intent semantics are still being preserved.

## Suites

```powershell
neurodrift evidence --suite synthetic --compact
```

Runs repeated controlled simulations across seeds and drift levels.

```powershell
neurodrift evidence --suite fixtures --compact
```

Runs available local fixtures. The included fixtures are smoke tests, not final
evidence.

```powershell
neurodrift evidence --suite all --figure-dir artifacts\evidence
```

Runs synthetic cases and local fixtures, then writes SVG figures.

## Report Fields

- `split_rate`: how often best decoder MSE and best meaning preservation
  disagree.
- `overalignment_rate`: how often an aligner improves MSE while worsening
  weak-intent meaning.
- `overalignment_penalty`: amount of meaning loss among over-aligned cases.
- `hard_label_loss`: ambiguity lost by collapsing weak intent into hard labels.
- `source_target_intent_shift`: weak intent movement between source and target.
- `best_mse_alignment_gain`: decoder gain for the method that wins by MSE.
- `best_meaning_distance`: semantic distance for the method that wins by
  meaning preservation.

Each scalar summary includes a deterministic bootstrap interval.

## What Would Matter

The synthetic suite is the machinery check. It shows the evaluator can detect
the failure mode.

The stronger result is:

- run the same suite on NLB-style neural population data
- run it on human invasive BCI data from DANDI when the labels support it
- compare against CEBRA, LFADS/NDT-style latents, and manifold stabilizers
- report whether ambiguity or geometry shift predicts later meaning failure

## What Would Weaken The Thesis

- MSE and meaning preservation always choose the same method.
- Hard-label ambiguity does not predict relabeling pressure.
- Geometry drift never separates from intent drift.
- Real neural sessions only show ordinary decoder degradation.

That is fine. A good thesis should be easy to attack.
